#!/usr/bin/env python
(nohup python -u scripts/exp_hong_al.py ebu3b sdh > nohup.hong.ebu3b.sdh; slack_notify --msg 'nohup.hong.ebu3b.sdh at labpc')
(nohup python -u scripts/exp_hong_al.py sdh ebu3b > nohup.hong.sdh.ebu3b; slack_notify --msg 'nohup.hong.sdh.ebu3b at labpc')
(nohup python -u scripts/exp_hong_al.py ebu3b ap_m > nohup.hong.ebu3b.ap_m; slack_notify --msg 'nohup.hong.ebu3b.ap_m at labpc')

(nohup python -u scripts/exp_hong_al.py ebu3b > nohup.hong.ebu3b; slack_notify --msg 'nohup.hong.ebu3b at labpc')
(nohup python -u scripts/exp_hong_al.py sdh > nohup.hong.sdh; slack_notify --msg 'nohup.hong.sdh at labpc')
(nohup python -u scripts/exp_hong_al.py ghc > nohup.hong.ghc; slack_notify --msg 'nohup.hong.ghc at labpc')
(nohup python -u scripts/exp_hong_al.py uva_cse > nohup.hong.uva_cse; slack_notify --msg 'nohup.hong.uva_cse at labpc')
