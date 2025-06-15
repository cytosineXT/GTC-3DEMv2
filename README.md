# install
```bash
git clone https://github.com/cytosineXT/GTC-3DEM.git
cd ./GTC-3DEM

conda create -n 3DEM python=3.9 -y
conda activate 3DEM
pip install -r requirements.txt

```

# train
```bash
python NNtrainGNN_arg4fold.py --cuda 'cpu'
```

when counter error about "RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.":


"/home/your_user_name/anaconda3/envs/3DEM/lib/python3.9/site-packages/torch/_tensor.py", line 1087

         if dtype is None:
             return self.numpy()
         else:
             return self.numpy().astype(dtype, copy=False)

modify to

         if dtype is None:
             return self.cpu().detach().numpy()
         else:
             return self.cpu().detach().numpy().astype(dtype, copy=False)

# inference
```bash
python NNval_GNN4foldbatch.py --pretrainweight 'output/testtrain/your_train_fold/your_trained_weight.pt'

```

# v1 
nn.embedding

# v2 
nn.linear embed

# v3 
v3.0 2025年6月12日20:31:08 PINN:helmholtz fft loss, helmloss without log
v3.1 helmloss with log
v3.2 2025年6月15日11:11:17 add reciporcity loss 
v3.3 2025年6月15日11:50:59 add pinnepoch, refine loss_fn