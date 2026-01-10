import veux


def plot_mode(model, numEigen, mode=1, scale=500.0):
    """
    This takes a structural model, and 
    """
    # First, get all node tags:
    nodeTags = model.getNodeTags()
    
    # This is a dictionary to hold mode shape 1
    modeShapes = []

    for i in range(numEigen):
        mode_dict = {}
        # Now, for each node tag, print the results
        for nodeTag in nodeTags:
            res = model.nodeEigenvector(nodeTag, i+1)
            # Add nodeTag and its displacement to the mode1 dictionary
            mode_dict[nodeTag] = res

        modeShapes.append(mode_dict)
    
    # For rendering purposes
    artist = veux.create_artist(model)
    artist.draw_outlines()
    # artist.draw_outlines(state=modeShapes[mode-1], scale=scale)
    #artist.draw_nodes()
    artist.draw_sections(state=modeShapes[mode-1], scale=scale)
    veux.serve(artist)

    pass


def plot_deformed_state(model, scale=500.0):
    """
    This takes a structural model, and 
    """
    # First, get all node tags:
    nodeTags = model.getNodeTags()
    
    # This dictionary will hold deformed state
    mode_dict = {}

    # Now, for each node tag, print the results
    for nodeTag in nodeTags:
        res = model.nodeDisp(nodeTag)
        # Add nodeTag and its displacement to the mode1 dictionary
        mode_dict[nodeTag] = res

    
    # For rendering purposes
    artist = veux.create_artist(model)
    artist.draw_outlines()
    # artist.draw_outlines(state=modeShapes[mode-1], scale=scale)
    #artist.draw_nodes()
    artist.draw_sections(state=mode_dict, scale=scale)
    veux.serve(artist)

    pass