# %%
import pandas as pd
import numpy as np
import paddle.fluid as fluid

def create_df(output):
    list = []
    for em in output:
        list.append(em[0])
    df = pd.DataFrame()
    df['similarity'] = list
    norm = pd.read_csv('../task_data/xw/norm_q.txt', sep='\t')
    df['norm_query'] = norm['text_a']
    df.sort_values('similarity', inplace=True, ascending=False)
    # print(df.head())
    # print(norm.head())
    return df

def cos_sim(np_x, np_y):
    x = fluid.layers.data(name='x', shape=[666, 768], dtype='float32', append_batch_size=False)
    y = fluid.layers.data(name='y', shape=[1, 768], dtype='float32', append_batch_size=False)
    out = fluid.layers.cos_sim(x, y)
    place = fluid.CUDAPlace(0)
    #place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    output = exe.run(feed={"x": np_x, "y": np_y}, fetch_list=[out])
    return output[0]
    # print(output)

# %%
if __name__ == '__main__':
    emb1 = np.load('norm/cls_emb.npy')
    emb2 = np.load('sim/cls_emb.npy')

    df_emb2 = pd.read_csv('../task_data/xw/sim_q.txt', sep='\t')
    i = 0
    for emb2_i in emb2:
        output = cos_sim(emb1, emb2_i)
        df = create_df(output)
        name = df_emb2['text_a'][i]
        print()
        i += 1
        df.to_csv('res/{}.txt'.format(name), index=False, sep='\t')
