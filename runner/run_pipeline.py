import os,sys,argparse,json,dataclasses
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.pipeline import run_for_client

def conv(obj):
    if dataclasses.is_dataclass(obj): return {k:conv(v) for k,v in dataclasses.asdict(obj).items()}
    if isinstance(obj,list): return [conv(x) for x in obj]
    if isinstance(obj,dict): return {k:conv(v) for k,v in obj.items()}
    return obj

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--client-id",default="CL001"); p.add_argument("--no-yfinance",action="store_true"); p.add_argument("--mc-paths",type=int,default=50000); args=p.parse_args()
    mandate,market,recs,candidates,top3,avg,ranking_meta=run_for_client(args.client_id,use_yfinance=not args.no_yfinance,mc_paths=args.mc_paths)
    os.makedirs("outputs/reports",exist_ok=True)
    out={"client":conv(mandate),"recommendations":conv(recs),"top3":conv(top3),"candidate_average":avg,"ranking_meta":ranking_meta}
    path=f"outputs/reports/{args.client_id}_results.json"; open(path,"w").write(json.dumps(out,indent=2,default=str)); print(json.dumps(out,indent=2,default=str)); print(f"\nSaved: {path}")
