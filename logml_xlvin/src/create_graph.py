import torch


def _action_to_one_hot(action, num_actions):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        action.size()[0], num_actions, dtype=torch.float32,
        device=action.device)
    return zeros.scatter_(1, action.unsqueeze(1), 1)


def create_graph(current_state, model, gamma, num_actions):
    '''
        inputs:
            current_state - tensor [1, 10]
            model         - TransE
        
        outputs:
            senders - tensor shape [number_of_edges]
            recievers - tensor shape [number_of_edges]
            features - tensor shape [number_of_edges, 9] (one-hot action(8) + gamma)
    '''

    #TODO: attention coefficients as actor
    #TODO: analyse scale of gnn output
    print('current_state', current_state.shape)

    device = current_state.device
    batch_size = current_state.shape[0]
    embed_dim = current_state.shape[-1]
    sq_num_actions = num_actions * num_actions

    first_children = torch.zeros(batch_size, num_actions, embed_dim, device=device)
    for action in range(num_actions):
        batched_action = torch.tensor([action], device=device).repeat(batch_size, 1)
        next_state = current_state + model.transition_model(current_state, batched_action)
        first_children[:, action, :] = next_state.detach()


    second_children = torch.zeros(batch_size, sq_num_actions, embed_dim, device=device)
    for action1 in range(num_actions):
        for action in range(num_actions):
            batched_action = torch.tensor([action], device=device).repeat(batch_size, 1)
            next_state = first_children[:, action1] + model.transition_model(first_children[:, action1], batched_action)
            second_children[:, action1 * num_actions + action, :] = next_state.detach()

    senders = []
    receivers = []
    node_features = []
    edge_features = []
    node_features += [current_state]   #add roots: 0, 1, 2, ...b-1

    for action in range(num_actions):
        # first children: (b, b+1, .. , b+7), (b+8, ..., b+15), ... (b+8(b-1), ..., 9b-1)
        receivers += [torch.tensor(i) for i in range(batch_size)]
        senders += [torch.tensor(batch_size + num_actions*i + action) for i in range(batch_size)]
        action = _action_to_one_hot(torch.tensor([action]), num_actions)
        gamma_tens = torch.ones(action.shape[0], 1) * gamma
        feat = torch.cat((action, gamma_tens), dim=-1)
        edge_features += [feat for i in range(batch_size)]

    node_features += [first_children.view(batch_size * num_actions, embed_dim)]

    for action1 in range(num_actions):
        for action2 in range(num_actions):
            # second children: (9b, 9b+1, ... , 9b+63), ..., (9b+64(b-1), ..., 73b-1)
            receivers += [torch.tensor(batch_size + num_actions*i + action2) for i in range(batch_size)]
            senders += [
                torch.tensor((num_actions+1)*batch_size + sq_num_actions*i + action1 * num_actions + action2)
                for i in range(batch_size)]
            action = _action_to_one_hot(torch.tensor([action2]), num_actions)
            gamma_tens = torch.ones(action.shape[0], 1) * gamma
            feat = torch.cat((action, gamma_tens), dim=-1)
            edge_features += [feat for i in range(batch_size)]

    node_features += [second_children.view(batch_size * sq_num_actions, embed_dim)]

    senders = torch.tensor(senders, device=device)
    receivers = torch.tensor(receivers, device=device)
    edge_features = torch.cat(edge_features, 0).to(device)
    node_features = torch.cat(node_features, 0).to(device)
    return node_features, senders, receivers, edge_features


def create_graph_v2(current_state, model, gamma, num_actions, num_steps=2, graph_detach=False):
    '''
        inputs:
            current_state - tensor [1, 10]
            model         - TransE
        
        outputs:
            senders - tensor shape [number_of_edges]
            recievers - tensor shape [number_of_edges]
            features - tensor shape [number_of_edges, 9] (one-hot action(8) + gamma)
    '''

    #TODO: analyse scale of gnn output

    device = current_state.device
    batch_size = current_state.shape[0]
    embed_dim = current_state.shape[-1]

    actions = torch.tensor([i for i in range(num_actions)], device=device)
    current_children = current_state.unsqueeze(1)
    children = [current_children]

    node_features = [current_state]
    edge_features = []
    senders = []
    receivers = []

    total_children = batch_size
    for i in range(num_steps):
        num_batches, num_children, _ = current_children.shape
        # stacked_children is (c1, c1, ..., c1, c2, c2, .., c2, ...)
        # each child repeated num_actions times + flattened for batches
        stacked_children = current_children.repeat((1, 1, num_actions))\
            .reshape((num_batches * num_actions * num_children, embed_dim))
        # stacked action is (0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, ....)
        # repeated enough times for each child to have its action
        stacked_actions = actions.repeat((num_batches, num_children))\
            .reshape((num_batches * num_children * num_actions))

        transitions = model.transition_model(stacked_children, stacked_actions)
        next_children = (stacked_children + transitions).reshape((num_batches, num_actions * num_children, embed_dim))
        if graph_detach:
            next_children = next_children.detach()
        children.append(next_children)
        current_children = next_children

        one_hot_action = _action_to_one_hot(stacked_actions, num_actions)
        gamma_tens = torch.ones(one_hot_action.shape[0], 1, device=device) * gamma
        edge_features.append(torch.cat([one_hot_action, gamma_tens], -1))

        node_features.append(next_children.view(num_batches * num_actions * num_children, embed_dim))
        senders.append(torch.range(total_children, total_children + batch_size * num_actions * num_children - 1,
                                   dtype=torch.int64))
        receivers.append(torch.range(total_children - num_batches * num_children,
                                     total_children - 1, dtype=torch.int64).repeat((num_actions, 1)).T.flatten())


        total_children += batch_size * num_actions * num_children
    
    node_features = torch.cat(node_features).to(device)
    edge_features = torch.cat(edge_features).to(device)
    senders = torch.cat(senders).to(device)   # should be 1, 2, 3, ..., last_child
    receivers = torch.cat(receivers).to(device)  # should be 0,0, ..0, 1, .., 1, 2, ..., 2
                                     # each repeated num_actions times
    return node_features, senders, receivers, edge_features
