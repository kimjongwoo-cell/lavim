"""nav4 base arm with VLMAS_NAV2 off (Planner emits no text; nav4 prompt keeps the fixed placeholder targets)."""
import hashlib
SRC = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/rtask_nav4_fulr_arm.sh"
DST = "/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/rtask_nav4_nonav2_arm.sh"
s = open(SRC).read()
assert hashlib.md5(s.encode()).hexdigest().startswith("fc4eac08"), hashlib.md5(s.encode()).hexdigest()[:8]

def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (old[:70], s.count(old))
    s = s.replace(old, new)

rep("""  fulr)      K=$ROOT/fulr;      ARMENV="VLMAS_RNLCR=fulr VLMAS_RNLCR_LOG=$ROOT/fulr/rnlcr_shard$SHARD.jsonl" ;;""",
    """  fulr)      K=$ROOT/fulr;      ARMENV="VLMAS_RNLCR=fulr VLMAS_RNLCR_LOG=$ROOT/fulr/rnlcr_shard$SHARD.jsonl" ;;
  nonav2)    K=$ROOT/nonav2;    ARMENV="" ;;   # nav4 base with VLMAS_NAV2 unset (see NAV2 line below)""")
# the only env difference: drop VLMAS_NAV2=1 for this arm
rep("""    VLMAS_NAV4=1 VLMAS_NAV2=1 VLMAS_CROSS_SCALE_ROUTER=0 \\""",
    """    VLMAS_NAV4=1 $NAV2ENV VLMAS_CROSS_SCALE_ROUTER=0 \\""")
rep("""CLAIM=${CLAIMDIR:-$ROOT/claims}""",
    """NAV2ENV="VLMAS_NAV2=1"; [ "$ARM" = nonav2 ] && NAV2ENV=""
CLAIM=${CLAIMDIR:-$ROOT/claims}""")
# smoke gate: this arm must show no Planner text decode
rep("""  fu=$(grep -c '^\\[RNLCR\\] case .* mode=fulr applied=True' $o.log); fs=$(grep -c '^\\[RNLCR.*SKIP' $o.log)""",
    """  fu=$(grep -c '^\\[RNLCR\\] case .* mode=fulr applied=True' $o.log); fs=$(grep -c '^\\[RNLCR.*SKIP' $o.log)
  pt=$(grep -c '^\\[PlanTargets\\]' $o.log); need_pt_zero=0; [ "$ARM" = nonav2 ] && need_pt_zero=1""")
rep("""     && [ "$fu" -eq "$need_fu" ] && [ "$fs" -eq 0 ]; then""",
    """     && [ "$fu" -eq "$need_fu" ] && [ "$fs" -eq 0 ] \\
     && { [ "$need_pt_zero" -eq 0 ] || [ "$pt" -eq 0 ]; }; then""")
rep("""    echo "PASS results=$r nav4=$n4 prunev4=$v4 eovc=$ev secld=$sl nova=$nv fulr=$fu fulr_skip=$fs empty=$em tb=$t" > $K/SMOKE_GATE_$ARM""",
    """    echo "PASS results=$r nav4=$n4 prunev4=$v4 eovc=$ev secld=$sl nova=$nv fulr=$fu fulr_skip=$fs plantargets=$pt empty=$em tb=$t" > $K/SMOKE_GATE_$ARM""")
rep("""    echo "FAIL results=$r nav4=$n4""", """    echo "FAIL plantargets=$pt results=$r nav4=$n4""")
s = s.replace("#   ARM=nova_fulr : nova arm + VLMAS_RNLCR=fulr",
              "#   ARM=nova_fulr : nova arm + VLMAS_RNLCR=fulr\n#   ARM=nonav2   : nav4 base WITHOUT VLMAS_NAV2 (Planner decodes no x5/x20 text; the nav4 prompt then carries\n#                  fixed_patch_plan's placeholder targets). Compare against runs/nav4/base on the same questions.", 1)
open(DST, "w").write(s)
print("wrote", DST, hashlib.md5(s.encode()).hexdigest()[:8])
