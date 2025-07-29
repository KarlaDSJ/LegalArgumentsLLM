from typing import List, Dict, Optional

import arguebuf as ag
from spacy.tokens import Span, Doc

from recap_am.model.config import config
from recap_am.relation.controller import mc_from_relations
from recap_am.relation.controller.attack_support import (
    Relation,
    RelationClass,
    classify,
)


def run(
    doc: Doc, relations: Optional[Dict[str, List[Relation]]], preset_mc: bool
) -> ag.Graph:
    adus = doc._.ADU_Sents
    claims = doc._.Claim_Sents
    premises = doc._.Premise_Sents
    mc = doc._.MajorClaim

    if config["adu"]["MC"]["method"] == "relations" and not preset_mc:
        mc = mc_from_relations.run_str(adus, relations)
        print("1")

    if not relations:
        relations = classify(adus)
        print("2")

    graph = ag.Graph(name=doc._.key.split("/")[-1])
    #    mc_node = ag.Node(graph.keygen(), mc, ag.NodeCategory.I, major_claim=True)
    mc_node=ag.AtomNode(mc)
    graph.major_claim=mc_node
    cnodes = []
    graph.add_node(mc_node)
    print("3")

    for claim in claims:
        if claim != mc:
            cnode = ag.AtomNode(claim)  
            print("cnode",cnode.label) 
            graph.add_node(cnode)         
            snode = _gen_snode(graph, relations[claim.text], mc)
            print("snode",snode)
            # cnode = ag.Node(graph.keygen(), claim, ag.NodeCategory.I)
            # snode = _gen_snode(graph, relations[claim.text], mc)

            if snode:
                cnodes.append(cnode)
                graph.add_node(snode)
                print("snode1",snode)

                graph.add_edge(ag.Edge(source=cnode, target=snode))
                graph.add_edge(ag.Edge(source=snode, target=mc_node))

    for premise in premises:
        if premise != mc:
            pnode = ag.AtomNode(premise)
            graph.add_node(pnode)
            match = (mc_node, 0)

            if cnodes:
                scores = {
                    cnode: min(
                        abs(cnode.text.start - pnode.text.end),
                        abs(cnode.text.end - pnode.text.start),
                    )
                    for cnode in cnodes
                }

                min_score = min(scores.values())
                candidates = [
                    node for node, score in scores.items() if score == min_score
                ]

                for candidate in candidates:
                    sim = pnode.text.similarity(candidate.text)
                    if sim > match[1]:
                        match = (candidate, sim)

            snode = _gen_snode(graph, relations[premise.text], match[0].text)
            print("premisse",snode.label)

            if snode:
                graph.add_node(snode)
                graph.add_edge(ag.Edge(source=pnode, target=snode))
                graph.add_edge(ag.Edge(source=snode, target=match[0]))

    return graph




def _gen_snode(
    graph: ag.Graph, relations: List[Relation], adu: Span
) -> Optional[ag.SchemeNode]:
    
    #candidates = []
    # print("relation",relations)
    # print("adu",adu.text)
    # for r in relations:
    #     # On observe les différences éventuelles entre r.adu et adu.text
    #     print(f"Comparaison : ADU => '{adu.text.strip()}' | rel.adu => '{r.adu.strip()}'")
    #     print("4")
    #     if r.adu.strip().lower() == adu.text.strip().lower():
    #         candidates.append(r)
    #         print("match",r)
    candidates = list(filter(lambda rel: rel.adu.strip() == adu.text.strip(), relations))
    
    if candidates:
        relation = candidates[0]

        if relation and relation.classification == RelationClass.ATTACK:
            print(relation.classification)
            return ag.SchemeNode(scheme=ag.Attack.DEFAULT)
            #return ag.Node(graph.keygen(), "Default Conflict", ag.NodeCategory.CA)
        elif relation and relation.classification == RelationClass.SUPPORT:
            print(relation.classification)
            return ag.SchemeNode(scheme= ag.Support.DEFAULT)
            #return ag.Node(graph.keygen(), "Default Inference", ag.NodeCategory.RA)

    return None
