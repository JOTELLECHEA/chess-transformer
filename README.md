### This repo is in progress, comments and writeup are being worked on 

This project trains on data derived from the Lichess open database, released under CC0. Games were filtered to GM-titled players across standard time controls and converted to UCI notation via the pipeline in src/


vocab_fixed.json is the canonical, closed-form move vocabulary; every model in this repo trains against this exact file