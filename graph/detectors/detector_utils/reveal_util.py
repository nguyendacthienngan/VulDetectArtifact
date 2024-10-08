from gensim.models import Word2Vec
import numpy as np
import torch
from torch_geometric.data import Data, Batch

import json
from typing import Dict, List, Tuple

from graph.detectors.models.reveal import type_map, model_args, ClassifyModel

class RevealUtil(object):
    def __init__(self, pretrain_model: Word2Vec, reveal_model: ClassifyModel, device: str):
        self.pretrain_model = pretrain_model
        self.reveal_model = reveal_model
        self.arrays = np.eye(len(type_map))
        self.device = device

    # Generates the initial embedding of each ASTNode in the graph. This function is only executed once during the training phase.
    # nodeContent[0] is type, nodeContent[1] is token sequence
    def generate_initial_astNode_embedding(self, nodeContent: List[str]) -> np.array:
        # type vector
        n_c = self.arrays[type_map[nodeContent[0]] - 1]
        # token sequence
        token_seq: List[str] = nodeContent[1].split(' ')
        # n_v = np.array([self.pretrain_model[word] if word in self.pretrain_model.wv.vocab else
        #                    np.zeros(model_args.vector_size) for word in token_seq]).mean(axis=0)
        # https://github.com/piskvorky/gensim/wiki/Migrating-from-Gensim-3.x-to-4
        keyed_vectors = self.pretrain_model.wv
        n_v = np.array([keyed_vectors.get_vector(word) if word in keyed_vectors.key_to_index else
                    np.zeros(keyed_vectors.vector_size) for word in token_seq]).mean(axis=0)
        
        v = np.concatenate([n_c, n_v])
        return v

    # Generate information for each AST initial node
    def generate_initial_node_info(self, ast: Dict) -> Data:
        astEmbedding: np.array = np.array([self.generate_initial_astNode_embedding(node_info) for node_info in ast["contents"]])
        x: torch.FloatTensor = torch.FloatTensor(astEmbedding)
        edges: List[List[int]] = [[edge[1], edge[0]] for edge in ast["edges"]]
        edge_index: torch.LongTensor = torch.LongTensor(edges).t()
        return Batch.from_data_list([Data(x=x, edge_index=edge_index)]).to(device=self.device)

    # Preprocess training data. The training process is only called once.
    def generate_initial_embedding(self, data: Dict) -> Tuple[int, List[Data], torch.LongTensor]:
        # label, List[ASTNode], edge
        if "target" in data.keys():
            label: int = data["target"]
        else:
            label: int = 0

        # edges
        cfgEdges = [json.loads(edge)[:2] for edge in data["cfgEdges"]]
        cdgEdges = [json.loads(edge) for edge in data["cdgEdges"]]
        ddgEdges = [json.loads(edge)[:2] for edge in data["ddgEdges"]]
        edges = cfgEdges + cdgEdges + ddgEdges
        edge_index: torch.LongTensor = torch.LongTensor(edges).t()

        # nodes
        nodes_info: List[Dict] = [json.loads(node_infos) for node_infos in data["nodes"]]
        graph_data_for_each_nodes: List[Data] = [self.generate_initial_node_info(node_info) for node_info in nodes_info]

        return (label, graph_data_for_each_nodes, edge_index)

    # Generate graph initialization vector, called once per epoch
    def generate_initial_graph_embedding(self, graph_info: Tuple[int, List[Data], torch.LongTensor]) -> Data:
        # self.reveal_model.embed_graph(data)[0] return graph_embedding for initial CPG node
        initial_embeddings: List[torch.FloatTensor] = [self.reveal_model.embed_graph(data.x, data.edge_index, None)[0].reshape(-1,)
                                                       # 某些AST子树可能没有子结点，直接取其值作为node embedding
                                                       if len(data.edge_index) > 0 else data.x[0]
                                                       for data in graph_info[1]]
        X: torch.FloatTensor = torch.stack(initial_embeddings)
        return Data(x=X, edge_index=graph_info[2], y=torch.tensor([graph_info[0]], dtype=torch.long))




import sys

if __name__ == '__main__':
    device = "cuda"
    w2v_model_path = sys.argv[1]
    pretrain_model = Word2Vec.load(w2v_model_path)
    reveal_model: ClassifyModel = ClassifyModel().to(device)
    reveal_util = RevealUtil(pretrain_model, reveal_model, device)

    sample = {
		"fileName":"CWE121_Stack_Based_Buffer_Overflow__char_type_overrun_memcpy_01.c",
		"cfgEdges":[
			"[0,1,\"\"]",
			"[1,2,\"\"]",
			"[2,3,\"\"]",
			"[3,4,\"\"]",
			"[4,5,\"\"]",
			"[5,6,\"\"]"
		],
		"nodes":[
			"{\"line\":37,\"edges\":[[0,1],[1,2],[1,3]],\"contents\":[[\"IdentifierDeclStatement\",\"charVoid structCharVoid ;\",\"\"],[\"IdentifierDecl\",\"structCharVoid\",\"\"],[\"IdentifierDeclType\",\"charVoid\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"]]}",
			"{\"line\":38,\"edges\":[[0,1],[1,2],[1,3],[2,4],[2,5],[3,6],[3,7]],\"contents\":[[\"ExpressionStatement\",\"structCharVoid . voidSecond = ( void * ) SRC_STR ;\",\"\"],[\"AssignmentExpr\",\"structCharVoid . voidSecond = ( void * ) SRC_STR\",\"=\"],[\"MemberAccess\",\"structCharVoid . voidSecond\",\"\"],[\"CastExpression\",\"( void * ) SRC_STR\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"voidSecond\",\"\"],[\"CastTarget\",\"void *\",\"\"],[\"Identifier\",\"SRC_STR\",\"\"]]}",
			"{\"line\":40,\"edges\":[[0,1],[1,2],[1,3],[2,4],[3,5],[5,6],[6,7],[6,8],[8,9],[8,10]],\"contents\":[[\"ExpressionStatement\",\"printLine ( ( char * ) structCharVoid . voidSecond ) ;\",\"\"],[\"CallExpression\",\"printLine ( ( char * ) structCharVoid . voidSecond )\",\"\"],[\"Callee\",\"printLine\",\"\"],[\"ArgumentList\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"Identifier\",\"printLine\",\"\"],[\"Argument\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"CastExpression\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"CastTarget\",\"char *\",\"\"],[\"MemberAccess\",\"structCharVoid . voidSecond\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"voidSecond\",\"\"]]}",
			"{\"line\":42,\"edges\":[[0,1],[1,2],[1,3],[2,4],[3,5],[3,6],[3,7],[5,8],[6,9],[7,10],[8,11],[8,12],[10,13],[10,14]],\"contents\":[[\"ExpressionStatement\",\"memcpy ( structCharVoid . charFirst , SRC_STR , sizeof ( structCharVoid ) ) ;\",\"\"],[\"CallExpression\",\"memcpy ( structCharVoid . charFirst , SRC_STR , sizeof ( structCharVoid ) )\",\"\"],[\"Callee\",\"memcpy\",\"\"],[\"ArgumentList\",\"structCharVoid . charFirst , SRC_STR , sizeof ( structCharVoid )\",\"\"],[\"Identifier\",\"memcpy\",\"\"],[\"Argument\",\"structCharVoid . charFirst\",\"\"],[\"Argument\",\"SRC_STR\",\"\"],[\"Argument\",\"sizeof ( structCharVoid )\",\"\"],[\"MemberAccess\",\"structCharVoid . charFirst\",\"\"],[\"Identifier\",\"SRC_STR\",\"\"],[\"SizeofExpr\",\"sizeof ( structCharVoid )\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"charFirst\",\"\"],[\"Sizeof\",\"sizeof\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"]]}",
			"{\"line\":43,\"edges\":[[0,1],[1,2],[1,3],[2,4],[2,5],[4,6],[4,7],[5,8],[5,9],[8,10],[8,11],[10,12],[10,13],[11,14],[11,15],[13,16],[13,17]],\"contents\":[[\"ExpressionStatement\",\"structCharVoid . charFirst [ ( sizeof ( structCharVoid . charFirst ) / sizeof ( char ) ) - 1 ] = '\\\\0' ;\",\"\"],[\"AssignmentExpr\",\"structCharVoid . charFirst [ ( sizeof ( structCharVoid . charFirst ) / sizeof ( char ) ) - 1 ] = '\\\\0'\",\"=\"],[\"ArrayIndexing\",\"structCharVoid . charFirst [ ( sizeof ( structCharVoid . charFirst ) / sizeof ( char ) ) - 1 ]\",\"\"],[\"CharExpression\",\"'\\\\0'\",\"\"],[\"MemberAccess\",\"structCharVoid . charFirst\",\"\"],[\"AdditiveExpression\",\"( sizeof ( structCharVoid . charFirst ) / sizeof ( char ) ) - 1\",\"-\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"charFirst\",\"\"],[\"MultiplicativeExpression\",\"sizeof ( structCharVoid . charFirst ) / sizeof ( char )\",\"/\"],[\"IntegerExpression\",\"1\",\"\"],[\"SizeofExpr\",\"sizeof ( structCharVoid . charFirst )\",\"\"],[\"SizeofExpr\",\"sizeof ( char )\",\"\"],[\"Sizeof\",\"sizeof\",\"\"],[\"MemberAccess\",\"structCharVoid . charFirst\",\"\"],[\"Sizeof\",\"sizeof\",\"\"],[\"SizeofOperand\",\"char\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"charFirst\",\"\"]]}",
			"{\"line\":44,\"edges\":[[0,1],[1,2],[1,3],[2,4],[3,5],[5,6],[6,7],[6,8],[8,9],[8,10]],\"contents\":[[\"ExpressionStatement\",\"printLine ( ( char * ) structCharVoid . charFirst ) ;\",\"\"],[\"CallExpression\",\"printLine ( ( char * ) structCharVoid . charFirst )\",\"\"],[\"Callee\",\"printLine\",\"\"],[\"ArgumentList\",\"( char * ) structCharVoid . charFirst\",\"\"],[\"Identifier\",\"printLine\",\"\"],[\"Argument\",\"( char * ) structCharVoid . charFirst\",\"\"],[\"CastExpression\",\"( char * ) structCharVoid . charFirst\",\"\"],[\"CastTarget\",\"char *\",\"\"],[\"MemberAccess\",\"structCharVoid . charFirst\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"charFirst\",\"\"]]}",
			"{\"line\":45,\"edges\":[[0,1],[1,2],[1,3],[2,4],[3,5],[5,6],[6,7],[6,8],[8,9],[8,10]],\"contents\":[[\"ExpressionStatement\",\"printLine ( ( char * ) structCharVoid . voidSecond ) ;\",\"\"],[\"CallExpression\",\"printLine ( ( char * ) structCharVoid . voidSecond )\",\"\"],[\"Callee\",\"printLine\",\"\"],[\"ArgumentList\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"Identifier\",\"printLine\",\"\"],[\"Argument\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"CastExpression\",\"( char * ) structCharVoid . voidSecond\",\"\"],[\"CastTarget\",\"char *\",\"\"],[\"MemberAccess\",\"structCharVoid . voidSecond\",\"\"],[\"Identifier\",\"structCharVoid\",\"\"],[\"Identifier\",\"voidSecond\",\"\"]]}"
		],
		"cdgEdges":[],
		"ddgEdges":[
			"[0,1,\"structCharVoid\"]",
			"[1,2,\"structCharVoid\"]",
			"[1,3,\"structCharVoid\"]",
			"[1,4,\"structCharVoid\"]",
			"[1,5,\"structCharVoid\"]",
			"[1,6,\"structCharVoid\"]",
			"[1,2,\"structCharVoid . voidSecond\"]",
			"[1,6,\"structCharVoid . voidSecond\"]"
		],
		"functionName":"CWE121_Stack_Based_Buffer_Overflow__char_type_overrun_memcpy_01_bad",
        "target": 1
	}

    graph_info: Tuple[int, List[Data], torch.LongTensor] = reveal_util.generate_initial_embedding(sample)
    data: Data = reveal_util.generate_initial_graph_embedding(graph_info)
    print(data)
    probs = reveal_model(data=Batch.from_data_list([data]).to(device))
    _, prediction = torch.max(probs, -1)
    ori_prob = torch.softmax(probs, dim=1)
    print(prediction)
    print(ori_prob)