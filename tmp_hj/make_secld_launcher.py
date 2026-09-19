"""Derive sender_relay_exp/rtask_nav4_secld_arm.sh from rtask_nav4_arm.sh (which is running -> never edited)."""
import sys
src = open("sender_relay_exp/rtask_nav4_arm.sh").read()
def rep(old, new, n=1):
    global src
    assert src.count(old) == n, (old[:60], src.count(old))
    src = src.replace(old, new)
rep("#   ARM=pruneb : + --variant pruning_b                (novelty + 0.25*texture + quota; §12 baseline)\n",
    "#   ARM=pruneb : + --variant pruning_b                (novelty + 0.25*texture + quota; §12 baseline)\n"
    "#   ARM=secld  : + --variant pruning_b + VLMAS_SECLD=1 (C1 #19 spatial evidence-coverage log-det)\n"
    "# (copy of rtask_nav4_arm.sh with the secld arm; same ROOT runs/nav4 and claim dir.)\n")
rep('  pruneb) K=$ROOT/pruneb; VARIANT=pruning_b; ARMENV="" ;;\n',
    '  pruneb) K=$ROOT/pruneb; VARIANT=pruning_b; ARMENV="" ;;\n'
    '  secld)  K=$ROOT/secld;  VARIANT=pruning_b; ARMENV="VLMAS_SECLD=1" ;;\n')
rep('eovc_md5=$(md5sum $REPO/memory/eovc.py | cut -c1-8)"',
    'eovc_md5=$(md5sum $REPO/memory/eovc.py | cut -c1-8) secld_md5=$(md5sum $REPO/memory/secld_select.py | cut -c1-8)"')
rep("eovc=$(grep -c '^\\[EOVC\\]' $out.log) prunev4=",
    "eovc=$(grep -c '^\\[EOVC\\]' $out.log) secld=$(grep -c '^\\[SECLD\\]' $out.log) empty_crops=$(grep -o 'empty_crops=[0-9]*' $out.log | grep -vc 'empty_crops=0') prunev4=")
rep("  ev=$(grep -c '^\\[EOVC\\]' $o.log); v4=$(grep -c 'PruneVision/v4' $o.log)\n",
    "  ev=$(grep -c '^\\[EOVC\\]' $o.log); v4=$(grep -c 'PruneVision/v4' $o.log); sl=$(grep -c '^\\[SECLD\\]' $o.log)\n")
rep("  need_ev=0; [ \"$ARM\" = eovc ] && need_ev=$v4\n",
    "  need_ev=0; [ \"$ARM\" = eovc ] && need_ev=$v4\n"
    "  need_sl=0; [ \"$ARM\" = secld ] && need_sl=$v4\n")
rep("  need_v4=0; { [ \"$ARM\" = eovc ] || [ \"$ARM\" = pruneb ]; } && need_v4=1\n",
    "  need_v4=0; { [ \"$ARM\" = eovc ] || [ \"$ARM\" = pruneb ] || [ \"$ARM\" = secld ]; } && need_v4=1\n")
rep("     && [ \"$ev\" -eq \"$need_ev\" ] && [ \"$okv4\" -eq 1 ]; then\n",
    "     && [ \"$ev\" -eq \"$need_ev\" ] && [ \"$sl\" -eq \"$need_sl\" ] && [ \"$okv4\" -eq 1 ]; then\n")
rep('    echo "PASS results=$r nav4=$n4 prunev4=$v4 eovc=$ev empty=$em tb=$t" > $K/SMOKE_GATE_$ARM',
    '    echo "PASS results=$r nav4=$n4 prunev4=$v4 eovc=$ev secld=$sl empty=$em tb=$t" > $K/SMOKE_GATE_$ARM')
rep('    echo "FAIL results=$r nav4=$n4 prunev4=$v4 eovc=$ev need_eovc=$need_ev need_prunev4=$need_v4 empty=$em tb=$t" > $K/SMOKE_GATE_$ARM',
    '    echo "FAIL results=$r nav4=$n4 prunev4=$v4 eovc=$ev secld=$sl need_eovc=$need_ev need_secld=$need_sl need_prunev4=$need_v4 empty=$em tb=$t" > $K/SMOKE_GATE_$ARM')
open("sender_relay_exp/rtask_nav4_secld_arm.sh", "w").write(src)
print("launcher written")
