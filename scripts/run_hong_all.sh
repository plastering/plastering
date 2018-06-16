#!/usr/bin/env python
(nohup python -u scripts/exp_hong_al.py ebu3b sdh > nohup.hong.ebu3b.sdh; slack_notify --msg 'hong ebu3b sdh at ozone') &
(nohup python -u scripts/exp_hong_al.py sdh ebu3b > nohup.hong.sdh.ebu3b; slack_notify --msg 'hong sdh ebu3b at ozone') &
(nohup python -u scripts/exp_hong_al.py ebu3b ap_m > nohup.hong.ebu3b.ap_m; slack_notify --msg 'hong ebu3b ap_m at ozone') &
