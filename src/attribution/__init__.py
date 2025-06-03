from .perturbation_based import PerturbationBasedAttribution
from .self_citation import SelfCitationAttribution
def create_attr(args, llm):
    if args.attr_type == 'tracllm' or args.attr_type == 'vanilla_perturb':
        attr = PerturbationBasedAttribution(llm,args.explanation_level,args.K,args.attr_type, args.score_funcs, args.sh_N,args.w,args.beta,args.verbose)
    elif args.attr_type == 'self_citation':
        attr = SelfCitationAttribution(llm, args.explanation_level,args.K,args.self_citation_model,args.verbose)
    else: raise NotImplementedError
    return attr