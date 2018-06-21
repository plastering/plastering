#!/usr/bin/env python
(nohup python -u scripts/exp_zodiac.py ebu3b sdh > nohup.zodiac.ebu3b.sdh; slack_notify --msg 'nohup.zodiac.ebu3b.sdh at lab-pc') &
(nohup python -u scripts/exp_zodiac.py sdh ebu3b > nohup.zodiac.sdh.ebu3b; slack_notify --msg 'nohup.zodiac.sdh.ebu3b at lab-pc') &
(nohup python -u scripts/exp_zodiac.py ebu3b ap_m > nohup.zodiac.ebu3b.ap_m; slack_notify --msg 'nohup.zodiac.ebu3b.ap_m') &

(nohup python -u scripts/exp_zodiac.py ebu3b > nohup.zodiac.ebu3b; slack_notify --msg 'zodiac ebu3b sdh at ozone') &
(nohup python -u scripts/exp_zodiac.py sdh > nohup.zodiac.sdh; slack_notify --msg 'nohup.zodiac.sdh') &
(nohup python -u scripts/exp_zodiac.py uva_cse > nohup.zodiac.uva_cse; slack_notify --msg 'nohup.zodiac.uva_cse') &
(nohup python -u scripts/exp_zodiac.py ghc > nohup.zodiac.ghc ; slack_notify --msg 'nohup.zodiac.ghc') &
