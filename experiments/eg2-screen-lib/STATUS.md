# eg2-screen-lib status (2026-10-13)

- survey.py: 60 captures bot17, all start_norah (stuck bot).
- Manual: 1 click (1050,100) dismissed Norah popup -> afk_popup.
  New edge proposal: start_norah --click(1050,100)--> afk_popup
  (start_norah has NO out-edges in game_world.yaml).
- autodiscover.py: 8 captures, all afk_popup, 0 unknowns.
- walker.py: afk_popup ESC-resistant (2x ESC, no change). 0 edges. Stop per rules.
- Completeness audit: 5 unmatchable screens (bag/menu/messages/hero/region_overview),
  9 screens with no out-edges. Descriptions complete (0 missing).
- Bot17 parked on afk_popup. Next needs Confirm click (commit action) — awaiting owner go.
