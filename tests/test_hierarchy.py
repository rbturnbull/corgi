from corgi.hierarchy import create_hierarchy


def test_create_hierarchy():
    options = [
        "A>B>C",
        "A>B>D",
        "A>E",
        "F>G>H",
    ]
    tree, classification_to_node, classification_to_node_id = create_hierarchy(options)

    assert tree.render_equal(
        """
        root
        ├── A
        │   ├── B
        │   │   ├── C
        │   │   └── D
        │   └── E
        └── F
            └── G
                └── H
        """        
    )
    for option in options:
        assert option in classification_to_node
        assert option in classification_to_node_id
    
    a_b_c = classification_to_node["A>B>C"]
    assert a_b_c.name == "C"
    assert a_b_c.parent.name == "B"
    assert a_b_c.parent.parent.name == "A"
    assert a_b_c.parent.parent.parent.name == "root"

    a_b_c_id = classification_to_node_id["A>B>C"]
    assert a_b_c_id == tree.node_to_id[a_b_c] == 4
