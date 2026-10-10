"""Direct-mode tests for ReefCommons.

Web is not used. LLM is mocked so vitality verdicts are deterministic.
"""

import json

CONTRACT = "contracts/reef_commons.py"
CHARTER = (
    "Support polyps that clearly improve the reef under the published charter. "
    "Reject empty, harmful, or non-contributing settlements."
)
ONE = 10**18


def deploy(direct_vm, direct_deploy, owner, thrive=700, min_mooring=ONE, tolerance=100):
    direct_vm.sender = owner
    direct_vm.value = 0
    return direct_deploy(
        CONTRACT,
        CHARTER,
        thrive,
        min_mooring,
        tolerance,
    )


def stake(direct_vm, amount):
    direct_vm.value = amount


def mock_settle(direct_vm, verdict="accept", vitality=800, notes="improves the reef"):
    direct_vm.mock_llm(
        r"neutral warden for an evolving reef commons",
        json.dumps({
            "verdict": verdict,
            "vitality_bps": vitality,
            "reasoning": notes,
        }),
    )


def mock_survey(direct_vm, verdict="retain", vitality=750, notes="still a net positive"):
    direct_vm.mock_llm(
        r"neutral warden re-surveying",
        json.dumps({
            "verdict": verdict,
            "vitality_bps": vitality,
            "reasoning": notes,
        }),
    )


def rejects(direct_vm, match):
    return direct_vm.expect_revert(match)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

def test_constructor_rejects_bad_thrive(direct_vm, direct_deploy, direct_owner):
    direct_vm.sender = direct_owner
    with rejects(direct_vm, "thrive_threshold_bps out of range"):
        direct_deploy(CONTRACT, CHARTER, 2000, 0, 0)


def test_constructor_rejects_bad_tolerance(direct_vm, direct_deploy, direct_owner):
    direct_vm.sender = direct_owner
    with rejects(direct_vm, "tolerance_bps out of range"):
        direct_deploy(CONTRACT, CHARTER, 600, 0, 2000)


def test_constructor_rejects_negative_mooring(direct_vm, direct_deploy, direct_owner):
    direct_vm.sender = direct_owner
    with rejects(direct_vm, "min_mooring must be >= 0"):
        direct_deploy(CONTRACT, CHARTER, 600, -1, 50)


def test_initial_state(direct_vm, direct_deploy, direct_owner):
    contract = deploy(direct_vm, direct_deploy, direct_owner)
    state = contract.get_reef_state()

    assert state["thrive_threshold_bps"] == 700
    assert state["toxic_threshold_bps"] == 700 // 3
    assert state["dissolve_threshold_bps"] == 700 // 2
    assert state["tolerance_bps"] == 100
    assert state["min_mooring"] == ONE
    assert state["polyp_count"] == 0
    assert state["thriving_count"] == 0
    assert state["bleached_count"] == 0
    assert state["dissolved_count"] == 0
    assert state["tide_pool"] == 0
    assert state["total_moored"] == 0
    assert contract.get_charter() == CHARTER


# ---------------------------------------------------------------------------
# seed_polyp
# ---------------------------------------------------------------------------

def test_seed_rejects_empty_substance(direct_vm, direct_deploy, direct_alice):
    contract = deploy(direct_vm, direct_deploy, direct_alice)
    stake(direct_vm, ONE)
    with rejects(direct_vm, "substance must not be empty"):
        contract.seed_polyp("", "   ")


def test_seed_rejects_below_min_mooring(direct_vm, direct_deploy, direct_alice):
    contract = deploy(direct_vm, direct_deploy, direct_alice)
    stake(direct_vm, ONE // 2)
    with rejects(direct_vm, "mooring below minimum"):
        contract.seed_polyp("", "A valid contribution to the reef.")


def test_seed_rejects_unknown_anchor(direct_vm, direct_deploy, direct_alice):
    contract = deploy(direct_vm, direct_deploy, direct_alice)
    stake(direct_vm, ONE)
    with rejects(direct_vm, "anchor_id does not exist"):
        contract.seed_polyp("polyp-999", "Anchored to nothing.")


def test_seed_thriving_creates_polyp_and_credits_catch(direct_vm, direct_deploy, direct_alice):
    contract = deploy(direct_vm, direct_deploy, direct_alice)
    stake(direct_vm, ONE * 2)
    mock_settle(direct_vm, "accept", 850, "clearly improves the reef")

    result = contract.seed_polyp("", "A detailed positive contribution to reef health.")

    assert result["thriving"] is True
    assert result["verdict"] == 1
    assert result["vitality_bps"] == 850
    assert result["polyp_id"] == "polyp-0"

    polyp = contract.get_polyp("polyp-0")
    assert polyp["condition"] == "thriving"
    assert polyp["vitality_bps"] == 850
    assert polyp["substance"].startswith("A detailed")
    assert polyp["survey_count"] == 0

    state = contract.get_reef_state()
    assert state["polyp_count"] == 1
    assert state["thriving_count"] == 1
    assert state["bleached_count"] == 0
    assert state["total_moored"] == ONE * 2

    # harvest is 5%, bloom bonus from empty tide pool is 0
    settler = polyp["settler"]
    assert contract.get_catch(settler) > 0


def test_seed_reject_increases_bleached_and_does_not_store(direct_vm, direct_deploy, direct_alice):
    contract = deploy(direct_vm, direct_deploy, direct_alice)
    stake(direct_vm, ONE)
    mock_settle(direct_vm, "reject", 100, "does not improve the reef")

    result = contract.seed_polyp("", "A weak or harmful settlement.")

    assert result["thriving"] is False
    assert result["polyp_id"] == ""

    state = contract.get_reef_state()
    assert state["polyp_count"] == 1          # counter still increments
    assert state["thriving_count"] == 0
    assert state["bleached_count"] == 1
    assert state["total_moored"] == ONE

    with rejects(direct_vm, "polyp does not exist"):
        contract.get_polyp("polyp-0")


def test_seed_low_vitality_forfeits_full_mooring(direct_vm, direct_deploy, direct_alice):
    contract = deploy(direct_vm, direct_deploy, direct_alice, thrive=600)
    stake(direct_vm, ONE)
    # toxic = 600 // 3 = 200
    mock_settle(direct_vm, "accept", 50, "toxic low vitality")

    contract.seed_polyp("", "Low quality settlement.")

    state = contract.get_reef_state()
    assert state["bleached_count"] == 1
    assert state["tide_pool"] == ONE          # full forfeit


def test_seed_mid_vitality_partial_refund(direct_vm, direct_deploy, direct_alice):
    contract = deploy(direct_vm, direct_deploy, direct_alice, thrive=600)
    stake(direct_vm, ONE * 2)
    # toxic = 200, so 300 is above toxic but below thrive
    mock_settle(direct_vm, "accept", 300, "partial quality")

    contract.seed_polyp("", "Mediocre settlement.")

    state = contract.get_reef_state()
    assert state["bleached_count"] == 1
    # half refunded to catch, half to tide pool
    assert state["tide_pool"] == ONE


def test_seed_with_existing_anchor(direct_vm, direct_deploy, direct_alice):
    contract = deploy(direct_vm, direct_deploy, direct_alice)
    stake(direct_vm, ONE)
    mock_settle(direct_vm, "accept", 800, "first polyp")
    first = contract.seed_polyp("", "Root polyp.")
    assert first["thriving"] is True

    direct_vm.clear_mocks()
    stake(direct_vm, ONE)
    mock_settle(direct_vm, "accept", 820, "anchored child")
    second = contract.seed_polyp("polyp-0", "Child polyp anchored to the root.")

    assert second["thriving"] is True
    assert contract.get_polyp("polyp-1")["anchor_id"] == "polyp-0"
    assert contract.get_reef_state()["thriving_count"] == 2


# ---------------------------------------------------------------------------
# survey_polyp
# ---------------------------------------------------------------------------

def test_survey_unknown_fails(direct_vm, direct_deploy, direct_alice):
    contract = deploy(direct_vm, direct_deploy, direct_alice)
    with rejects(direct_vm, "polyp does not exist"):
        contract.survey_polyp("polyp-99")


def test_survey_non_thriving_fails(direct_vm, direct_deploy, direct_alice):
    contract = deploy(direct_vm, direct_deploy, direct_alice)
    stake(direct_vm, ONE)
    mock_settle(direct_vm, "reject", 100, "rejected")
    contract.seed_polyp("", "Rejected settlement.")

    with rejects(direct_vm, "polyp is not thriving"):
        contract.survey_polyp("polyp-0")


def test_survey_retain_updates_vitality_and_count(direct_vm, direct_deploy, direct_alice):
    contract = deploy(direct_vm, direct_deploy, direct_alice)
    stake(direct_vm, ONE)
    mock_settle(direct_vm, "accept", 800, "initial")
    contract.seed_polyp("", "Thriving polyp.")

    direct_vm.clear_mocks()
    mock_survey(direct_vm, "retain", 780, "still good")
    result = contract.survey_polyp("polyp-0")

    assert result["dissolved"] is False
    assert result["verdict"] == 1
    assert result["vitality_bps"] == 780
    assert result["survey_count"] == 1

    polyp = contract.get_polyp("polyp-0")
    assert polyp["condition"] == "thriving"
    assert polyp["vitality_bps"] == 780
    assert polyp["survey_count"] == 1
    assert contract.get_reef_state()["dissolved_count"] == 0


def test_survey_retract_dissolves(direct_vm, direct_deploy, direct_alice):
    contract = deploy(direct_vm, direct_deploy, direct_alice)
    stake(direct_vm, ONE)
    mock_settle(direct_vm, "accept", 800, "initial")
    contract.seed_polyp("", "Thriving polyp.")

    direct_vm.clear_mocks()
    mock_survey(direct_vm, "retract", 200, "no longer positive")
    result = contract.survey_polyp("polyp-0")

    assert result["dissolved"] is True
    assert result["verdict"] == 0

    polyp = contract.get_polyp("polyp-0")
    assert polyp["condition"] == "dissolved"
    assert contract.get_reef_state()["dissolved_count"] == 1

    with rejects(direct_vm, "polyp is not thriving"):
        contract.survey_polyp("polyp-0")


def test_survey_low_vitality_dissolves_even_on_retain(direct_vm, direct_deploy, direct_alice):
    contract = deploy(direct_vm, direct_deploy, direct_alice, thrive=600)
    # dissolve_threshold = 300
    stake(direct_vm, ONE)
    mock_settle(direct_vm, "accept", 800, "initial")
    contract.seed_polyp("", "Thriving polyp.")

    direct_vm.clear_mocks()
    mock_survey(direct_vm, "retain", 250, "dropped below dissolve line")
    result = contract.survey_polyp("polyp-0")

    assert result["dissolved"] is True
    assert contract.get_polyp("polyp-0")["condition"] == "dissolved"


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def test_list_polyps_pagination(direct_vm, direct_deploy, direct_alice):
    contract = deploy(direct_vm, direct_deploy, direct_alice)

    for i in range(3):
        stake(direct_vm, ONE)
        mock_settle(direct_vm, "accept", 800 + i, f"polyp {i}")
        contract.seed_polyp("", f"Substance for polyp {i}.")
        direct_vm.clear_mocks()

    page = contract.list_polyps(0, 2)
    assert page["total"] == 3
    assert page["offset"] == 0
    assert len(page["polyps"]) == 2
    assert page["polyps"][0]["id"] == "polyp-0"
    assert page["polyps"][1]["id"] == "polyp-1"

    page2 = contract.list_polyps(2, 10)
    assert len(page2["polyps"]) == 1
    assert page2["polyps"][0]["id"] == "polyp-2"


def test_get_catch_unknown_address_is_zero(direct_vm, direct_deploy, direct_alice):
    contract = deploy(direct_vm, direct_deploy, direct_alice)
    assert contract.get_catch("0x" + "00" * 20) == 0