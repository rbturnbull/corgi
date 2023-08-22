from typing import List, Tuple
from hierarchicalsoftmax import SoftmaxNode

def create_hierarchy(classifications:List[str], label_smoothing:float=0.0, gamma:float=0.0) -> Tuple[SoftmaxNode, dict]:
    classification_tree = SoftmaxNode(name="root", label_smoothing=label_smoothing, gamma=gamma)
    classification_to_node = {}
    for classification in classifications:
        components = classification.split(">")
        current_node = classification_tree
        for component in components:
            child_names = [child.name for child in current_node.children]
            if component in child_names:
                current_node = current_node.children[child_names.index(component)]
            else:
                current_node = SoftmaxNode(name=component, parent=current_node, label_smoothing=label_smoothing, gamma=gamma)
            

        classification_to_node[classification] = current_node
        
    classification_tree.set_indexes()
    classification_to_node_id = {
        classification: classification_tree.node_to_id[node] 
        for classification, node in classification_to_node.items()
    }
    return classification_tree, classification_to_node, classification_to_node_id
