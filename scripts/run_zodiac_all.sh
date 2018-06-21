#!/usr/bin/env python
(nohup python -u scripts/exp_zodiac.py ebu3b sdh > nohup.zodiac.ebu3b.sdh; slack_notify --msg 'zodiac ebu3b sdh at ozone') &
(nohup python -u scripts/exp_zodiac.py sdh ebu3b > nohup.zodiac.sdh.ebu3b; slack_notify --msg 'zodiac sdh ebu3b at ozone') &
(nohup python -u scripts/exp_zodiac.py ebu3b ap_m > nohup.zodiac.ebu3b.ap_m; slack_notify --msg 'zodiac ebu3b ap_m at ozone') &

(nohup python -u scripts/exp_zodiac.py ebu3b sdh > nohup.zodiac.ebu3b.sdh; slack_notify --msg 'zodiac ebu3b sdh at ozone') &
(nohup python -u scripts/exp_zodiac.py sdh ebu3b > nohup.zodiac.sdh.ebu3b; slack_notify --msg 'zodiac sdh ebu3b at ozone') &
(nohup python -u scripts/exp_zodiac.py ebu3b ap_m > nohup.zodiac.ebu3b.ap_m; slack_notify --msg 'zodiac ebu3b ap_m at ozone') &
