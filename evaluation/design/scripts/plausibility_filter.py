"""A-priori metallurgical plausibility filter for candidate compositions.

A composition can satisfy every property target a physics model checks and
still be something no one could melt, cast or forge. Solid-solution models
reward adding solutes, so an unconstrained search drifts toward compositions
carrying every permitted element at once. This filter encodes constraints that
hold for real superalloys and rejects candidates outside them.

DERIVATION. Each rule states a qualitative principle from the metallurgical
literature and takes its numeric bound from the empirical envelope of the 77
commercial alloys in the knowledge graph -- the same alloys the system is built
on, spanning wrought, cast, DS and single-crystal grades from Nimonic 75 through
CMSX-10. The bounds are deliberately set at the extreme of observed practice,
not at a typical value, so the filter rejects only what lies outside the range
of alloys that are actually manufactured. It is calibrated against real alloys,
never against the search results it scores.

Reed, R.C., "The Superalloys: Fundamentals and Applications", Cambridge
University Press (2006), is cited below as [Reed2006] for the principles;
chapter 2 covers alloy chemistry and the role of each element class.

WHERE THE BOUNDS DIFFER FROM THE ORIGINAL SPECIFICATION. Three of the four
proposed thresholds were tighter than industrial practice and would have
rejected real production alloys. Each was relaxed to the observed maximum:

  proposed                     observed in real alloys        adopted
  <= 6 elements > 1 wt%        max 8 (Rene N4 SC, Rene 125)   <= 8
  refractory <= 14 wrought     max 16.0 (Haynes 230)          <= 16
  refractory <= 20 cast        max 20.0 (WAZ-20 DS)           <= 20  (unchanged)
  Cu, Mn, Si each <= 0.5       Mn max 0.70, Si max 2.80       Cu <= 0.5,
                                                              Mn, Si <= 1.0

A <= 6 element rule would have rejected 12% of the commercial alloys, and a
14 wt% wrought refractory cap rejects Haynes 230, so both would have measured
conformity to an invented standard rather than to practice.
"""

#: Elements counted as refractory for the solid-solution / TCP budget.
REFRACTORY = ("Mo", "W", "Re", "Ta", "Nb")

#: Elements that appear in turbine-grade superalloys only as residuals.
TRACE_ONLY = {"Cu": 0.5, "Mn": 1.0, "Si": 1.0}

#: Threshold above which an element counts as a deliberate alloying addition.
MAJOR_THRESHOLD = 1.0

#: Rule bounds, all at the observed maximum of the 77-alloy reference set.
MAX_MAJOR_ELEMENTS = 8
MAX_REFRACTORY = {"wrought": 16.0, "cast": 20.0}
MAX_AL_TI_TA = {"wrought": 10.0, "cast": 14.5}


def _is_wrought(processing):
    return "wrought" in (processing or "").lower() or "forged" in (processing or "").lower()


def check_plausibility(composition, processing="cast"):
    """Return (passes, [failed rule names], detail dict).

    Rules, each with its principle and its numeric source:

    R1 element count -- Superalloy chemistry is a balance struck between a
       small number of deliberate additions; every extra element adds a phase
       the designer must control [Reed2006 ch.2]. No commercial alloy in the
       reference set carries more than 8 additions above 1 wt%.

    R2 refractory budget -- Mo, W, Re, Ta and Nb give solid-solution and
       creep strength but drive sigma/mu/Laves TCP precipitation, and the
       tolerable total is set by processing route: wrought alloys must stay
       hot-workable, castings can carry more [Reed2006 ch.2, ch.5].
       Observed maxima: 16.0 wrought (Haynes 230), 20.0 cast (WAZ-20 DS).

    R3 trace elements -- Cu, Mn and Si are not strengthening additions in
       turbine-grade superalloys; they enter as melt residuals and are held
       down because they degrade oxidation resistance and grain-boundary
       cohesion [Reed2006 ch.2]. Cu appears in none of the 77 reference
       alloys. Mn reaches 0.70 and Si 2.80, the latter only in Haynes HR-160,
       a deliberately silicided sulfidation-resistant grade rather than a
       turbine structural alloy; the 1.0 wt% bound excludes it by design.

    R4 gamma-prime former budget -- Al, Ti and Ta set the gamma-prime volume
       fraction, and the practical ceiling differs by route: wrought alloys
       are limited by hot workability, castings are not [Reed2006 ch.2].
       Observed maxima: 9.6 wrought, 14.3 cast.
    """
    comp = {k: float(v) for k, v in (composition or {}).items() if v}
    route = "wrought" if _is_wrought(processing) else "cast"
    failed, detail = [], {}

    majors = [e for e, v in comp.items() if e != "Ni" and v > MAJOR_THRESHOLD]
    detail["n_major"] = len(majors)
    if len(majors) > MAX_MAJOR_ELEMENTS:
        failed.append("R1_element_count")

    refr = sum(comp.get(e, 0.0) for e in REFRACTORY)
    detail["refractory"] = round(refr, 2)
    if refr > MAX_REFRACTORY[route]:
        failed.append("R2_refractory_budget")

    over = {e: comp.get(e, 0.0) for e, lim in TRACE_ONLY.items() if comp.get(e, 0.0) > lim}
    detail["trace_violations"] = over
    if over:
        failed.append("R3_trace_elements")

    alttata = comp.get("Al", 0.0) + comp.get("Ti", 0.0) + comp.get("Ta", 0.0)
    detail["al_ti_ta"] = round(alttata, 2)
    if alttata > MAX_AL_TI_TA[route]:
        failed.append("R4_gamma_prime_formers")

    return (not failed), failed, detail


def composition_stats(composition):
    """Descriptors used to characterise what each search arm produces."""
    comp = {k: float(v) for k, v in (composition or {}).items() if v}
    return {
        "n_major": sum(1 for e, v in comp.items() if e != "Ni" and v > MAJOR_THRESHOLD),
        "refractory": round(sum(comp.get(e, 0.0) for e in REFRACTORY), 2),
        "has_cu_mn_si": any(comp.get(e, 0.0) > lim for e, lim in TRACE_ONLY.items()),
    }
