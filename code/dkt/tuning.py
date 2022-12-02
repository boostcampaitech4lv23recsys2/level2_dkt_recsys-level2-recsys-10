import easydict
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def objective_function(space):
    """
    space 예시 {'batch_size': 64, 'lr': 0.00010810929882981193, 'n_layers': 1}
    """

    # args가 dict으로 건네지기 때문에 easydict으로 바꿔준다
    args = space['args']
    args = easydict.EasyDict(args)
    
    # 하이퍼파라메타 값 변경
    args.n_layers = space['n_layers']
    args.n_heads = space['n_heads']
    args.hidden_dim = space['hidden_dim']
    args.max_seq_len = space['seq_len']

    # seed 설정 
    seed_everything(args.seed)
    model = space['model']
    kf_auc = space['kf_auc']
    report = space['report']

    run(args, train_data, valid_data, model, report, kf_auc)

    best_auc = report['best_auc']

    return -best_auc


def trials_to_df(trials, space, best):
    # 전체 결과
    rows = []
    keys = list(trials.trials[0]['misc']['vals'].keys())

    # 전체 실험결과 저장
    for trial in trials:
        row = {}

        # tid
        tid = trial['tid']
        row['experiment'] = str(tid)
        
        # hyperparameter 값 저장
        vals = trial['misc']['vals']
        hparam = {key: value[0] for key, value in vals.items()}

        # space가 1개 - 값을 바로 반환
        # space가 다수 - dict에 값을 반환
        hparam = hyperopt.space_eval(space, hparam)

        if len(keys) == 1:
            row[keys[0]] = hparam
        else:
            for key in keys:
                row[key] = hparam[key]

        # metric
        row['metric'] = abs(trial['result']['loss'])
        
        # 소요 시간
        row['time'] = (trial['refresh_time'] - trial['book_time']).total_seconds() 
        
        rows.append(row)

    experiment_df = pd.DataFrame(rows)
    
    # best 실험
    row = {}
    best_hparam = hyperopt.space_eval(space, best)

    if len(keys) == 1:
        row[keys[0]] = best_hparam
    else:
        for key in keys:
            row[key] = best_hparam[key]
    row['experiment'] = 'best'

    best_df = pd.DataFrame([row])

    # best 결과의 auc / time searching 하여 찾기
    search_df = pd.merge(best_df, experiment_df, on=keys)
    
    # column명 변경
    search_df = search_df.drop(columns=['experiment_y'])
    search_df = search_df.rename(columns={'experiment_x': 'experiment'})

    # 가장 좋은 metric 결과 중 가장 짧은 시간을 가진 결과를 가져옴 
    best_time = search_df.time.min()
    search_df = search_df.query("time == @best_time")

    df = pd.concat([experiment_df, search_df], axis=0)

    return df