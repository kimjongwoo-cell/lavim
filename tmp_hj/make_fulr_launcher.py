"""Build sender_relay_exp/rtask_nav4_fulr_arm.sh from rtask_nav4_nova_arm.sh: arms fulr (nav4 base + FULR) and
nova_fulr (nav4 + NOVA + FULR), same ROOT runs/nav4 and claim dir, gate checks the FULR marker."""
import hashlib, sys
src_p, dst_p = sys.argv[1], sys.argv[2]
src = open(src_p).read()


def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (old[:70], src.count(old))
    src = src.replace(old, new)


rep('''#   ARM=nova   : --variant base + VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova KEEP=0.25 (C1 #21 NOVA, post-vision, + C1META bridge)
''', '''#   ARM=nova   : --variant base + VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova KEEP=0.25 (C1 #21 NOVA, post-vision, + C1META bridge)
#   ARM=fulr      : nav4 base + VLMAS_RNLCR=fulr (C2 #18 FULR: question-grounded latent formation + sink-norm-conserved read)
#   ARM=nova_fulr : nova arm + VLMAS_RNLCR=fulr
# (copy of rtask_nav4_nova_arm.sh with the two FULR arms; memory/rnlcr.py mode fulr, engine unchanged.)
''')
rep('''  nova)   K=$ROOT/nova;   ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1" ;;
''', '''  nova)   K=$ROOT/nova;   ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1" ;;
  fulr)      K=$ROOT/fulr;      ARMENV="VLMAS_RNLCR=fulr VLMAS_RNLCR_LOG=$ROOT/fulr/rnlcr_shard$SHARD.jsonl" ;;
  nova_fulr) K=$ROOT/nova_fulr; ARMENV="VLMAS_C1_QASC=1 VLMAS_C1_SELECT=nova VLMAS_NOVA_KEEP=0.25 VLMAS_C1_META_BRIDGE=1 VLMAS_RNLCR=fulr VLMAS_RNLCR_LOG=$ROOT/nova_fulr/rnlcr_shard$SHARD.jsonl" ;;
''')
rep(''' qasc_md5=$(md5sum $REPO/memory/qasc_select.py | cut -c1-8) used=$(mem $G)"''',
    ''' qasc_md5=$(md5sum $REPO/memory/qasc_select.py | cut -c1-8) rnlcr_md5=$(md5sum $REPO/memory/rnlcr.py | cut -c1-8) lu_md5=$(md5sum $REPO/memory/lu.py | cut -c1-8) engine_md5=$(md5sum $REPO/vision_text_mas/latent_qwen_engine.py | cut -c1-8) env=[$ARMENV] used=$(mem $G)"''')
rep(''' empty=$(grep -c 'EMPTY terminal answer' $out.log) tb=$(grep -c Traceback $out.log) oom=$(grep -ci 'out of memory' $out.log) eff=$(find $out -name efficiency.json | wc -l)"
}''', ''' fulr=$(grep -c '^\\[RNLCR\\] case .* mode=fulr applied=True' $out.log) fulr_skip=$(grep -c '^\\[RNLCR.*SKIP' $out.log) empty=$(grep -c 'EMPTY terminal answer' $out.log) tb=$(grep -c Traceback $out.log) oom=$(grep -ci 'out of memory' $out.log) eff=$(find $out -name efficiency.json | wc -l)"
}''')
rep('''  need_nv=0; [ "$ARM" = nova ] && need_nv=2
''', '''  need_nv=0; { [ "$ARM" = nova ] || [ "$ARM" = nova_fulr ]; } && need_nv=2
  fu=$(grep -c '^\\[RNLCR\\] case .* mode=fulr applied=True' $o.log); fs=$(grep -c '^\\[RNLCR.*SKIP' $o.log)
  need_fu=0; { [ "$ARM" = fulr ] || [ "$ARM" = nova_fulr ]; } && need_fu=2
''')
rep('''     && [ "$ev" -eq "$need_ev" ] && [ "$sl" -eq "$need_sl" ] && [ "$nv" -eq "$need_nv" ] && [ "$okv4" -eq 1 ]; then
    echo "PASS results=$r nav4=$n4 prunev4=$v4 eovc=$ev secld=$sl nova=$nv empty=$em tb=$t" > $K/SMOKE_GATE_$ARM
  else
    echo "FAIL results=$r nav4=$n4 prunev4=$v4 eovc=$ev secld=$sl nova=$nv need_nova=$need_nv need_eovc=$need_ev need_secld=$need_sl need_prunev4=$need_v4 empty=$em tb=$t" > $K/SMOKE_GATE_$ARM
''', '''     && [ "$ev" -eq "$need_ev" ] && [ "$sl" -eq "$need_sl" ] && [ "$nv" -eq "$need_nv" ] && [ "$okv4" -eq 1 ] \\
     && [ "$fu" -eq "$need_fu" ] && [ "$fs" -eq 0 ]; then
    echo "PASS results=$r nav4=$n4 prunev4=$v4 eovc=$ev secld=$sl nova=$nv fulr=$fu fulr_skip=$fs empty=$em tb=$t" > $K/SMOKE_GATE_$ARM
  else
    echo "FAIL results=$r nav4=$n4 prunev4=$v4 eovc=$ev secld=$sl nova=$nv fulr=$fu fulr_skip=$fs need_nova=$need_nv need_fulr=$need_fu need_eovc=$need_ev need_secld=$need_sl need_prunev4=$need_v4 empty=$em tb=$t" > $K/SMOKE_GATE_$ARM
''')
open(dst_p, "w").write(src)
print("wrote", dst_p, hashlib.md5(src.encode()).hexdigest()[:8])
