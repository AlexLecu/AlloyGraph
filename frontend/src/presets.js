// Reference compositions, shared by the composition editor and the evaluator's
// default state so there is one source of truth for them.
//
// These are real, published alloys. The evaluator previously opened on
// Ni 60 / Cr 20 / Al 10 / Ti 5 / Co 5 -- it sums to 100 but carries 15 wt% of
// gamma-prime formers, roughly half again the most any wrought superalloy
// contains, so the tool's first impression was a composition it would itself
// flag as implausible.
export const BUILTIN_PRESETS = {
  "Waspaloy": { composition: {"Ni": 58.0, "Cr": 19.5, "Co": 13.5, "Mo": 4.3, "Al": 1.3, "Ti": 3.0, "C": 0.08, "B": 0.006, "Zr": 0.06}, processing: "wrought", builtin: true },
  "Inconel 718": { composition: { "Ni": 52.5, "Cr": 19.0, "Fe": 19.0, "Nb": 5.1, "Mo": 3.0, "Ti": 0.9, "Al": 0.5 }, processing: "wrought", builtin: true },
  "Udimet 720": { composition: { "Ni": 55.0, "Cr": 16.0, "Co": 14.7, "Ti": 5.0, "Al": 2.5, "Mo": 3.0, "W": 1.25 }, processing: "wrought", builtin: true },
  "IN738LC": { composition: {"Ni": 61.5, "Cr": 16.0, "Co": 8.5, "Mo": 1.75, "W": 2.6, "Al": 3.4, "Ti": 3.4, "Ta": 1.75, "Nb": 0.9, "C": 0.11, "B": 0.01, "Zr": 0.05}, processing: "cast", builtin: true },
  "Udimet 500": { composition: { "Ni": 54.0, "Cr": 18.0, "Co": 18.5, "Mo": 4.0, "Al": 2.9, "Ti": 2.9, "C": 0.08, "B": 0.006, "Zr": 0.05 }, processing: "wrought", builtin: true },
  "Haynes 282": { composition: { "Ni": 57.0, "Cr": 19.5, "Co": 10.0, "Mo": 8.5, "Ti": 2.1, "Al": 1.5, "Fe": 1.0, "Mn": 0.15, "Si": 0.1, "C": 0.06, "B": 0.005 }, processing: "wrought", builtin: true },
  "CMSX-4": { composition: {"Ni": 61.7, "Cr": 6.5, "Co": 9.0, "Mo": 0.6, "W": 6.0, "Al": 5.6, "Ti": 1.0, "Ta": 6.5, "Re": 3.0, "Hf": 0.1}, processing: "cast", builtin: true },
  "Rene 65": { composition: {"Ni": 51.6, "Cr": 16.0, "Co": 13.0, "Mo": 4.0, "W": 4.0, "Al": 2.1, "Ti": 3.7, "Nb": 0.7, "Fe": 1.0, "B": 0.016, "Zr": 0.05, "C": 0.01}, processing: "wrought", builtin: true }
}

export const DEFAULT_PRESET = "Waspaloy"
