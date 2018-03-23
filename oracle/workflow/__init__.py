

f_class_dict = {
    'zodiac': ZodiacFramework,
    'scrabble': ScrabbleFramework
}


class Node():
    """docstring for Node"""
    def __init__(self, f, nexts=[]):
        self.f = f
        self.nexts = nexts

    def add_next(self, n):
        self.nexts.append(n)


        
    

class Workflow(object):
    """docstring for Workflow"""
    def __init__(self, f_graph_configs):
        """
        inputs
        f_names (dict): graph of the frameworks (name + configuration)
            ex: {
                    "zodiac": ({}, {
                        "scrabble": ({}, [])
                    })
                }
        """
        super(Workflow, self).__init__()
        self.arg = arg
        self.f_graph = self.traverse_dict(

    def traverse_config_dict(f_graph_configs):
        nexts = []
        for f_name, (f_config, f_nexts) in f_graph_configs.items():
            f = f_class_dict[f_name](f_config)
            curr_node = Node(f)
            curr_node.nexts = self.traverse_config_dict(f_nexts)
            nexts.append(curr_node)
        return nexts

if __name__ == '__main__':
    zodiac_config = {}
    scrabble_config = {}
    f_graph = {
        'zodiac': (zodiac_config, {
            'scrabble': (scrabble_config, [])
        })}
    worflow = Worflow(f_graph)
