import sys

import src.auto_short_sim as autoss
import src.charge_sequences as chseq
import src.data as dt


def main():
    args = sys.argv[1:]
    if len(args) == 0:
        print('Error: No mode supplied')
        return
    mode = args[0]
    if mode == 'predefined':
        if len(args) != 3:
            raise ValueError('usage: predefined [charge_seq_name] [repeat]')
        charge_seq_name = args[1]
        charge_seq = chseq.predefined.get(charge_seq_name)
        if charge_seq is None:
            raise ValueError('1st arg: charge sequence name not predefined')
        try:
            repeat = int(args[2])
        except ValueError:
            raise ValueError('2nd arg: repeat is not an integer')
    elif mode == 'random':
        raise ValueError('sorry, random mode not yet implemented')
    else:
        raise ValueError('invalid mode')
    auto_sim(charge_seq, charge_seq_name, repeat)


def auto_sim(charge_seq, charge_seq_name, repeat):
    config = autoss.get_params('data_cnn/config/params.cfg')
    ref_config = dt.load_ref_config('data_cnn/config/ref_config.npy')

    model_att = ['spring_len', 'spring_const', 'atom_radius', 'epsilon', 'boltzmann_const', 'temperature']
    model_params = {param: config[param] for param in model_att}

    sim_short_att = ['max_dist', 'trial_no', 'save_period', 'num_repeat']
    sim_short_params = {param: config[param] for param in sim_short_att}
    for att in sim_short_att:
        if att != 'max_dist':
            sim_short_params[att] = int(sim_short_params[att])

    autoss.repeat_short_sim(
        charge_seq=charge_seq,
        charge_seq_name=charge_seq_name,
        model_params=model_params,
        ref_config=ref_config,
        sim_short_params=sim_short_params,
        repeat=repeat,
        save_dir='data_cnn/data_raw',
    )


if __name__ == '__main__':
    main()
