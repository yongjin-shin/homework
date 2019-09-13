#! /bin/sh

# for i in 1 2 3;
# do echo $i;
# done

#for bb in 1000 5000 10000;
#do
#for lrr in 0.001 0.005 0.01 0.05;
#do python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b $bb -lr $lrr -rtg --exp_name hc_b${bb}_r${lrr}
#done
#done

rm data/lb* data/sb* -r

python train_pg_f19.py CartPole-v0 -n 100 -b 1000 -e 3 -dna --exp_name sb_no_rtg_dna
python train_pg_f19.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg -dna --exp_name sb_rtg_dna
python train_pg_f19.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg --exp_name sb_rtg_na
python train_pg_f19.py CartPole-v0 -n 100 -b 5000 -e 3 -dna --exp_name lb_no_rtg_dna
python train_pg_f19.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg -dna --exp_name lb_rtg_dna
python train_pg_f19.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg --exp_name lb_rtg_na

