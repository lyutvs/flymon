# M0d literature evidence — APL, ORN→PN gain control, KC excitability, glomerular bias

Compiled 2026-09-17. Every row was checked against text I actually downloaded (curl → text, or PMC/eLife HTML). "full-text" = I read the article body (publisher HTML, PMC author manuscript, or author PDF). Numbers are only those printed in the text; anything that appears only in a figure graphic is marked "figure-only / not stated in text". HTML-to-text conversion loses fraction bars and superscripts; where I restored them (e.g. Rm/Ra, ORN^1.5) this is noted.

Local copies of the converted texts: `m0d-lit/src/*.txt` (same scratchpad).

## Sources opened

| key | citation | DOI / URL opened | access |
|---|---|---|---|
| Amin2020 | Amin, Apostolopoulou, Suárez-Grimalt, Vrontou, Lin. Localized inhibition in the Drosophila mushroom body. eLife 9:e56954 | 10.7554/eLife.56954 — https://elifesciences.org/articles/56954 | full-text incl. Methods + Appendix 1 |
| Papa2011 | Papadopoulou, Cassenaer, Nowotny, Laurent. Normalization for sparse encoding of odors by a wide-field interneuron. Science 332:721 | 10.1126/science.1201835 — https://pmc.ncbi.nlm.nih.gov/articles/PMC3242050/ | full-text (author manuscript main text). Online Supplement NOT opened (PMC proof-of-work / science.org blocked) |
| Lin2014 | Lin, Bygrave, de Calignon, Lee, Miesenböck. Sparse, decorrelated odor coding in the MB enhances learned odor discrimination. Nat Neurosci 17:559 | 10.1038/nn.3660 — author manuscript PDF https://eprints.whiterose.ac.uk/id/eprint/89890/ | full-text |
| Inada2017 | Inada, Tsuchimoto, Kazama. Origins of cell-type-specific olfactory processing in the Drosophila MB circuit. Neuron 95:357 | 10.1016/j.neuron.2017.06.039 — https://kazamalab.riken.jp/pdf/Neuron_Inada_2017.pdf | full-text (publisher PDF on lab site) |
| Prisco2021 | Prisco, Deimel, Yeliseyeva, Fiala, Tavosanis. The APL neuron normalizes odour-evoked activity in the Drosophila MB calyx. eLife 10:e74172 | 10.7554/eLife.74172 — https://elifesciences.org/articles/74172 | full-text |
| Ray2020 | Ray, Aldworth, Stopfer. Feedback inhibition and its control in an insect olfactory circuit. eLife 9:e53281 (+ ModelDB 262670 code) | 10.7554/eLife.53281 — https://elifesciences.org/articles/53281 ; https://github.com/ModelDBRepository/262670 (mb/mod/gradedsyn.mod, mb/network/config.yaml) | full-text + model code |
| Takemura2017 | Takemura et al. A connectome of a learning and memory center in the adult Drosophila brain. eLife 6:e26975 | 10.7554/eLife.26975 — https://elifesciences.org/articles/26975 | full-text incl. Table 5 (parsed from HTML table) |
| Li2020 | Li et al. The connectome of the adult Drosophila mushroom body provides insights into function. eLife 9:e62576 | 10.7554/eLife.62576 — https://elifesciences.org/articles/62576 | full-text (figure graphics not parsed) |
| Larva2023 | MBON-a1/a2 define an odor intensity channel … larval Drosophila. Front Physiol 2023 | 10.3389/fphys.2023.1111244 — https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2023.1111244/full | full-text (side evidence, larva) |
| Olsen2010 | Olsen, Bhandawat, Wilson. Divisive normalization in olfactory population codes. Neuron 66:287 | 10.1016/j.neuron.2010.04.009 — https://pmc.ncbi.nlm.nih.gov/articles/PMC2866644/ | full-text (author manuscript) |
| OlsenWilson2008 | Olsen & Wilson. Lateral presynaptic inhibition mediates gain control in an olfactory circuit. Nature 452:956 | 10.1038/nature06864 — https://pmc.ncbi.nlm.nih.gov/articles/PMC2824883/ | full-text (author manuscript; Supplementary not opened) |
| Kazama2008 | Kazama & Wilson. Homeostatic matching and nonlinear amplification at identified central synapses. Neuron 58:401 | 10.1016/j.neuron.2008.02.030 — https://pmc.ncbi.nlm.nih.gov/articles/PMC2429849/ | full-text (author manuscript; Supplemental Procedures not opened) |
| Nagel2015 | Nagel, Hong, Wilson. Synaptic and circuit mechanisms promoting broadband transmission of olfactory stimulus dynamics. Nat Neurosci 18:56 | 10.1038/nn.3895 — https://pmc.ncbi.nlm.nih.gov/articles/PMC4289142/ | full-text (author manuscript incl. Methods) |
| Liu2021 | Liu, Li, Tang, Qin, Tu. Short-term plasticity regulates both divisive normalization and adaptive responses in Drosophila olfactory system. Front Comput Neurosci 15:730431 | 10.3389/fncom.2021.730431 — Europe PMC full-text XML of PMC8568954 | full-text |
| Abdel2021 | Abdelrahman, Vasilaki, Lin. Compensatory variability in network parameters enhances memory performance in the Drosophila mushroom body. **PNAS 118:e2102158118 (not Nature Communications)** | 10.1073/pnas.2102158118 — Europe PMC full-text XML of PMC8670477 | full-text (SI Appendix not opened) |
| ApoLin2020 | Apostolopoulou & Lin. Mechanisms underlying homeostatic plasticity in the Drosophila mushroom body in vivo. PNAS 117:16606 | 10.1073/pnas.1921294117 — https://pmc.ncbi.nlm.nih.gov/articles/PMC7368247/ | full-text (SI not opened) |
| Gruntman2013 | Gruntman & Turner. Integration of the olfactory code across dendritic claws of single MB neurons. Nat Neurosci 16:1821 | 10.1038/nn.3547 — https://pmc.ncbi.nlm.nih.gov/articles/PMC3908930/ | full-text (author manuscript) |
| Zheng2022 | Zheng et al. Structured sampling of olfactory input by the fly mushroom body. Curr Biol 32:3334 | 10.1016/j.cub.2022.06.031 — https://pmc.ncbi.nlm.nih.gov/articles/PMC9413950/ | full-text (author manuscript; supplementary tables not opened) |
| Caron2013 | Caron, Ruta, Abbott, Axel. Random convergence of olfactory inputs in the Drosophila mushroom body. Nature 497:113 | 10.1038/nature12063 — https://pmc.ncbi.nlm.nih.gov/articles/PMC4148081/ | full-text (author manuscript; Supplementary Figs not opened) |

---

## Q1. APL is non-spiking, graded and local

### Table 1a — Amin et al. 2020 (eLife 56954)

| claim | verbatim quote | source | numbers/constants | access |
|---|---|---|---|---|
| (a) APL is non-spiking | "receive feedback inhibition from a non-spiking interneuron called the anterior paired lateral (APL) neuron" | Amin2020 abstract | — | full-text |
| (a) low voltage-gated Na/Ca channel expression | "voltage-gated Na+ and Ca2+ channels are expressed at lower levels in APL than in all other types of mushroom body neurons" | Amin2020 Results (Fig. 2) | RNA-seq from Aso 2019; 21 MB cell types | full-text |
| (a) but release is voltage-dependent | "indicating that Ca2+ influx in APL is in large part voltage-dependent" | Amin2020 Results (Ort/histamine, Fig. 2—fs1) | — | full-text |
| (a) graded-release sensitivity (cited, not measured here) | "as little as 2 mV depolarization can modulate neurotransmitter release in non-spiking insect interneurons (Burrows and Siegler, 1978)" | Amin2020 Discussion | 2 mV (locust, cited) | full-text |
| (b) activity does not cross lobes 2–3 µm apart | "APL neurites in the unresponsive lobe failed to respond despite being only 2–3 µm away from responding neurites" | Amin2020 Results (Fig. 3, dTRPA1 in KC subsets) | 2–3 µm | full-text |
| (b) no spread across the whole cell | "local stimulation evoked intense GCaMP6f responses at the site of ATP application, but not at the unstimulated site (Figure 4A,D), indicating that activity in APL attenuates to undetectable levels across the breadth of the neuron (about 250–300 µm" | Amin2020 Results (Fig. 4, P2X2+ATP) | 250–300 µm breadth | full-text |
| (b) decay distance of APL activity | "APL activity decreased to zero by 100 µm from the ejection site and there was no activity in the horizontal lobe" | Amin2020 Results (vertical-lobe tip stimulation) | ≤100 µm | full-text |
| (b) best-fit space constant | "Overall, 50 µm gave the best fit: 25 µm did not spread activity enough, while 75 µm spread activity too far" | Amin2020 Results (Fig. 8F) | λ = 50 µm (tested 25/50/75/95/448) | full-text |
| (b) λ from literature Rm/Ra is too long | "suggesting that APL's normalized space constant is shorter than 95 µm and that RmRa might be unusually low in APL" | Amin2020 Results | HS-cell ratio 0.0907 m → λ≈95 µm; PN ratio 2.0 m → λ≈448 µm | full-text |
| (b) implied Rm/Ra | "(RmRa = 0.025 m for λ = 50 µm)" [slash lost in HTML] | Amin2020 Discussion | Rm/Ra = 0.025 m | full-text |
| (b) λ is a normalized (radius-dependent) constant | "we modeled space constants as varying with the square root of the neurite radius" | Amin2020 Results/Methods | λ=50 µm ↔ k = 0.1117 m^1/2; λ/k = 4.48×10^-4 m^1/2 | full-text |
| (b) kernel used | "w…=e-d(xi,k1,yj,k2)/λ" (Eq. 2) and "sk1,k2=∑i=1m∑j=1nw(xi,k1,yj,k2)" (Eq. 1) | Amin2020 Methods | s(k1,k2)=Σ_i Σ_j exp(−d/λ) over KC1→APL and APL→KC2 synapse pairs | full-text |
| (b) inhibitory effect spreads further than APL activity, but weaker | "the inhibitory effect on Kenyon cells persisted into the peduncle and the horizontal lobe … even into the calyx (>200 µm away) … However, this more extended inhibitory effect on the calyx was significantly smaller than the local inhibitory effect" | Amin2020 Results | >200 µm; no amplitude ratio stated in text (figure-only) | full-text |
| (b) stimulation resolution caveat | "decay to half-strength ~10–25 µm" (co-ejected red dye) | Amin2020 Discussion | 10–25 µm half-decay | full-text |
| (c) self > other inhibition, magnitude | "disappears entirely if the space constant is infinite (at λ = 50 µm, the median imbalance is ~40%)" | Amin2020 Results | median self/other ≈ 1.4 at λ=50 µm | full-text |
| (c) how measured (model prediction, not a recording) | "the violin plots show, for each KC1, the ratio of s(k1,k1) (self-inhibition) vs. the average of s(k1,k2) across all k2 (inhibiting other KCs)" | Amin2020 Fig. 8 legend | n = 1923 KCs, hemibrain v1.1; ~10,000 random points on skeleton for λ fitting | full-text |
| (c) connectome counts used | "All 1927 traced Kenyon cells … form reciprocal synapses with APL (49.6 ± 17.9 APL-KC and 52.6 ± 13.4 KC-APL synapses per KC; mean ± s.d.)" | Amin2020 Results | 95,678 APL→KC and 101,430 KC→APL synapses; skeleton 80.2 mm | full-text |
| (c) holds for calyx-only APL→KC synapses | "taking into account only APL-KC synapses in the calyx. Again, our model predicts that each Kenyon cell inhibits itself more than it inhibits other individual Kenyon cells on average" | Amin2020 Results (Fig. 8J,M) | except αβ-p (few calyx APL synapses) | full-text |
| (c) lateral inhibition still dominates in total | "given that there are ~2000 Kenyon cells, the sum total of lateral inhibition that an individual Kenyon cell receives would still be stronger than its own self-inhibition" | Amin2020 Discussion | ~2000 KCs | full-text |
| (c) self-inhibition ≈ gain control | "this effect of self-inhibition is better thought of, not as decorrelation per se, but rather as gain control" | Amin2020 Discussion | — | full-text |
| (c) shuffle control | "the imbalance disappeared when we shuffled the identities of which Kenyon cell each KC-APL or APL-KC synapse belonged to" | Amin2020 Results | — | full-text |
| (d) nonlinearity/saturation of APL release | No statement found ("saturat", "graded", "nonlinear" absent from the text); model explicitly omits it: "We ignored certain complicating factors … (e.g. active conductances; differences in synaptic strength or membrane conductance across APL) … (e.g. dynamic effects of feedback inhibition" | Amin2020 Discussion | not stated | full-text |
| (e) adult APL pre/post synapses everywhere | "The larval APL has pre-synapses only in the calyx, while adult APL has pre-synapses everywhere" | Amin2020 Discussion | — | full-text |
| (e) calyx vs lobe asymmetry | "stimulating Kenyon cells at the tip of the vertical lobe activated APL locally, but not in the calyx" ; "the ratio of (response in unstimulated site)/(response in stimulated) site is higher for calyx stimulation than lobe stimulation" | Amin2020 Results / Fig. 4 legend | p = 0.004 | full-text |
| (e) calyx inhibition silences KC axons | "sufficiently strong local inhibition in Kenyon cell dendrites can silence Kenyon cells throughout their axons" | Amin2020 Results | — | full-text |
| (e) proposed dual role | "enforcing Kenyon cell sparse coding in the calyx, and modulating learning in the compartments of the lobes" | Amin2020 Discussion | — | full-text |
| (e) neurite geometry | "these neurites average ~0.5 µm in diameter throughout APL; ~95% of neurite length is less than 1 µm in diameter and the maximum diameter is ~3 µm" | Amin2020 Discussion | 0.5 / <1 / 3 µm | full-text |
| MBON consequence is a prediction, untested | "Local inhibition of Kenyon cell output predicts that activity of MBONs near the site of APL activation would be more strongly inhibited than MBONs far away. This prediction may be tested in future experiments" | Amin2020 Discussion | — | full-text |

### Table 1b — Papadopoulou 2011, Lin 2014, Inada 2017, Prisco 2021, Ray 2020

| claim | verbatim quote | source | numbers/constants | access |
|---|---|---|---|---|
| GGN non-spiking, resting Vm | "GGN, is a non-spiking neuron with a resting potential of −51 ± 5 mV" | Papa2011 | −51 ± 5 mV; 80 recordings / 55 animals | full-text (main) |
| GGN graded response scales with odor concentration | "depolarization grew with stimulus concentration (tested over a million-fold) with a peak depolarization of 15 - 20 mV above rest" | Papa2011 | 15–20 mV peak | full-text (main) |
| GGN integrates global KC drive | "LFP (power) and VGGN (∫Vdt) co-varied over this concentration range (n = 364 pairs, linear fit, r = 0.93" | Papa2011 | r = 0.93 | full-text (main) |
| unitary KC→GGN EPSP | "Unitary EPSPs were 1 ± 0.50 mV (n = 11 KCs), with some nearing 2 mV" | Papa2011 | 1 ± 0.5 mV | full-text (main) |
| threshold-like onset of GGN→KC inhibition | "In every pair, GGN depolarization beyond 5mV reduced current-evoked firing of the recorded KC" | Papa2011 | ~5 mV | full-text (main) |
| graded release incl. tonic release at rest | "Note enhancement of LFP, indicating graded release of GABA at rest" (hyperpolarizing GGN) | Papa2011 Fig. 4 legend | — | full-text (main) |
| GGN itself is inhibited (gain of the loop is modulated) | "GGN is itself reciprocally connected to a spiking inhibitory interneuron" (IG) | Papa2011 | — | full-text (main) |
| GGN as population integrator | "the membrane potential of GGN can be thought of as equivalent to the PSTH of a population of spiking interneurons, smoothed with an EPSP-like kernel" | Papa2011 | — | full-text (main) |
| GGN can silence MB output | "it can, on its own, shut down entirely the output of the mushroom body" | Papa2011 | — | full-text (main) |
| Nonlinear fit form of GGN effect | Fig. 4E "Relationship and fit between LFP or β-lobe neuron outputs and depolarizing current injected in GGN. Note the steeper action on β-lobe neuron." | Papa2011 | fit function and parameters not stated in main text | full-text (main); SI not opened |
| all-to-all functional feedback (Drosophila) | "Because blocking output from all Kenyon cells is required to suppress inhibition in any lobe, feedback is in all likelihood all-to-all" | Lin2014 | class sizes "about 1/2, 1/6 and 1/3" (αβ, α′β′, γ) | full-text |
| APL block broadens/correlates KC codes | "population sparseness decreased and inter-odor correlations increased when either Kenyon cell or APL synaptic output were blocked" | Lin2014 | numeric sparseness values figure-only | full-text |
| inhibition scales with KC drive | "The increase was greater for the IA:EB mixtures than for δ-DL (Fig. 6a), supporting the idea that inhibitory feedback is driven by overall Kenyon cell activity" | Lin2014 | ORN totals 2,030 / 1,860 / 286 spikes/s (IA/EB/δ-DL); modeled PN 1,310 / 1,238 / 937 | full-text |
| coding level | "only ~5–10% of Kenyon cells respond to any given odor" | Lin2014 Intro (cited) | 5–10 % | full-text |
| APL local vs global depends on input strength | "responded to odors either locally within a lobe or globally across all lobes depending on the strength of stimuli" | Inada2017 Summary | β′ lobe responds at 10^-9–10^-7 dilution; β, γ only at higher | full-text |
| APL does not spike in the local regime | "APL neuron does not seem to spike within an input range that evokes localized GCaMP signals in only one of the lobes" | Inada2017 Discussion | — | full-text |
| local lateral inhibition, some KCs spared | "Some KCs were almost not hyperpolarized at all, suggesting that the inhibition is effective locally" | Inada2017 Results (Fig. 3E) | n = 19 KCs, k-means 2 groups | full-text |
| single-KC self/feedback inhibition, cell-type specific | "activation of single KCs recruits a hyperpolarizing offset response, but unexpectedly, only prominently in α′/β′ KCs" | Inada2017 Results (Fig. 4) | linear range 3–15 spikes in 500 ms | full-text |
| APL→KC strength equal across KC types, GABAA+GABAB | "responses were mediated by both GABAA and GABAB receptors … and, critically, similar in strength between all types of KCs" | Inada2017 Results (Fig. 5) | — | full-text |
| inhibition is slow | "Inhibition followed excitation by several hundred ms" | Inada2017 Discussion | several hundred ms | full-text |
| APL in calyx scales with PN input and is regional | "APL response scales with the PN input strength and is regionalized around PN input distribution" | Prisco2021 abstract | Δ(Oct−Mch) = 45 ± 27 %, Δ(Oct−δ-DL) = 76 ± 30 % | full-text |
| APL targets PN boutons too (presynaptic inhibition possible) | "Most of the APL-to-PN connections were localized on PN boutons (84% ± 2%" ; "a presynaptic component of APL inhibition is certainly possible" | Prisco2021 | 126/136 calyx PNs reciprocally connected to APL; r² = 0.63 (PN), 0.60 (KC) reciprocal-count correlation | full-text |
| APL normalizes claw responses; block → input-like variability | "blocking APL output led to a more variable odour representation at the level of KC claws" | Prisco2021 | — | full-text |
| gradient model | "APL inhibition onto MGs can be imagined as a gradient that peaks at the MGs active during a given stimulus and attenuates with distance" | Prisco2021 Discussion | no λ stated | full-text |
| GGN compartmental model: attenuation α-lobe→calyx | "depolarizations of ~10 mV in GGN's α lobe branch do indeed attenuate greatly with distance to amplitudes as low as 5 mV in parts of the calyx" | Ray2020 Discussion | RA = 100 Ωcm, RM = 33 kΩcm² (Fig. 2 legend) | full-text |
| Graded, saturating (sigmoid) GGN→KC synapse in model | code: "sinf = 1 / (1 + exp((vmid - vpre) / vslope))", "g = gbar * s", "s' = (sinf - s) / tau" | Ray2020 ModelDB gradedsyn.mod | — | model code |
| sigmoid constants used | config.yaml: "vmid: -40.0mV # -40mV in Papadopoulou et al 2011"; "vslope: 5.0mV"; "gmax: 0.7nS # 50nS in Papadopoulou, et al., 2011"; "tau: 4.0ms"; "e: -80mV # -90 mV in Papadopoulou et al., 2011" | Ray2020 config.yaml | vmid −40 mV, slope 5 mV, τ 4 ms, E −80 mV, gmax 0.7 nS per KC (lognormal, std = mean) | model code; attribution to Papa2011 SI NOT verified |
| inhibition expands KC dynamic range | "our model also predicts that feedback inhibition from GGN can expand the dynamic range of inputs able to activate KCs" | Ray2020 Discussion | — | full-text |
| weak stimuli → local, strong → global (speculation) | "Weak stimuli, on the other hand, might recruit only local inhibition" | Ray2020 Discussion | — | full-text |

Derived (my arithmetic, not a source statement): with vmid = −40 mV, slope = 5 mV and GGN rest −51 mV, s∞(rest) = 1/(1+e^{11/5}) ≈ 0.10 and at +20 mV (−31 mV) ≈ 0.86, i.e. the model release saturates within the measured 15–20 mV odor range.

### Not found / uncertain (Q1)
- Amin 2020 gives no release nonlinearity, no saturation, no APL input–output curve, no synaptic time constants. λ = 50 µm is a calcium-imaging fit to a passive exponential kernel (normalized for neurite radius); the paper states it is likely an overestimate of spread.
- The "~40%" is a connectome-model prediction (median s(k1,k1)/mean s(k1,k2) at λ = 50 µm), not a direct physiological measurement. Values at other λ are figure-only.
- Papadopoulou 2011 Online Supplement (SI 1 simulations, SI 5 calibration, SI 7) could not be downloaded; the sigmoid constants (−40 mV, 5 mV, 50 nS, 4 ms, −90 mV) are only attributed to it in Ray 2020's config comments, so they are unverified.
- No dedicated computational model of *local* Drosophila APL (distributed APL with a spatial kernel in a spiking network) was found beyond Amin 2020's phenomenological kernel; Prisco 2021 is experimental (no model). A GitHub student project (sachinksalim/drosophilia_mushroom_body) exists but is not peer-reviewed and was not used.
- Lin 2014 numeric sparseness/correlation values are figure-only.

---

## Q2. APL outputs onto MBONs and DANs

### Table 2

| claim | verbatim quote | source | numbers/constants | access |
|---|---|---|---|---|
| α lobe: APL has essentially no DAN or MBON output except MBON-α1 | "In fact, APL sends no output to any DANs and no MBONs other than a few connections to MBON-α1 (Table 5)" | Takemura2017 | Table 5 (APL presynaptic): MBON-α1-A 29, MBON-α1-B 34; MBON-α2/α3 0; PPL1-α3-A/B 0; PPL1-α′2α2-A/B 0; PAM-α1 (16) 0 | full-text |
| interpretation | "Thus, APL's role seems largely confined to influencing the sensory input the KCs convey to the lobes" | Takemura2017 | — | full-text |
| APL↔KC counts in α lobe | Table 5 row "KCs (α lobe)": APL post-synaptic 9128, pre-synaptic 4123 | Takemura2017 Table 5 | KC→APL 9128; APL→KC 4123 (α lobe); by compartment APL→KC α3 1443, α2 1276, α1 1404 | full-text |
| APL output onto feedforward MBON axon in α lobe | Table 5 row "MBON-β1>α": APL post 2, pre 76 | Takemura2017 Table 5 | APL→MBON-β1>α 76 (its axon in α lobe; text does not discuss) | full-text |
| DANs → APL (input, not output) | Table 5 rows PPL1-α3-A/B "10 / 13", PPL1-α′2α2-A/B "38 / 45", PAM-α1 (16) "21" (APL post-synaptic) | Takemura2017 Table 5 | — | full-text |
| APL branches are separate units in α lobe | "these individual branches are not connected inside the α lobe (Video 11). These processes may each serve as discrete units of local inhibitory feedback in the lobes" | Takemura2017 | — | full-text |
| hemibrain paper: MBONs receive APL input, no counts in text | "Within the MB lobes, MBONs also receive input from APL (Liu and Davis, 2009) and DPM (Waddell et al., 2000) as well as from DANs (Takemura et al., 2017)" | Li2020 | APL→MBON / APL→DAN counts not stated in text | full-text (figures not parsed) |
| hemibrain: APL is a major calyx input to KCs | "The next most prominent inputs are KC-to-KC synapses within the CA (11.9%), from APL (9.5%" | Li2020 Fig. 3—fs legend | APL = 9.5 % of KC input in main calyx; 9.1 % vACA; 10.9 % dACA; 0.9 % lACA | full-text |
| locust: GGN effect on output (β-lobe) neurons is indirect | "The action of GGN on the LN was not direct, for GGN had no effect on LN firing evoked by current injection (SI 7)" ; "GGN affects LNs indirectly by its actions on the KC population output" | Papa2011 | n = 10 LNs | full-text (main) |
| larva (side evidence): APL→calyx MBON synapses exist | "APL has 14 inputs (4% of the total) to MBON-a1-R" ; "We found APL to MBON-a1/a2 connections only in the calyx and not around the lobes" | Larva2023 | 14 / 8 (right), 10 / 6 (left) presynaptic sites | full-text |

### Not found / uncertain (Q2)
- No adult-Drosophila functional study was found showing *direct* APL→MBON inhibition (e.g. APL activation/silencing changing MBON current-evoked responses). Amin 2020 frames MBON effects as a KC-mediated prediction still to be tested; in locust, Papadopoulou 2011 found the GGN effect on output neurons to be indirect.
- Li 2020 does not report APL→MBON or APL→DAN synapse counts in text (at most figure graphics/neuPrint). I could not verify from literature that MBON05 (γ4>γ1γ2) is APL's largest MBON target; that is a MaleCNS-model fact, not a literature fact. Takemura 2017 covers only the α lobe (no γ lobe MBONs).

---

## Q3. ORN→PN gain control

### Table 3

| claim | verbatim quote | source | numbers/constants | access |
|---|---|---|---|---|
| intraglomerular transfer (Eq. 1) | "PN=Rmax(ORN1.5ORN1.5+σ1.5)" [= Rmax·ORN^1.5/(ORN^1.5+σ^1.5); superscripts lost] | Olsen2010 | exponent 1.5 | full-text |
| per-glomerulus fits | "Rmax = 170, 167, 163, and 144, and σ = 16.3, 11.8, 12.4, and 44.8, for glomeruli DM4, DL5, VM7, and DM1, respectively" | Olsen2010 Methods | spikes/s | full-text |
| Rmax, σ glomerulus-invariant | "Rmax and σ are essentially the same for all glomeruli (except that without GABA receptor antagonists σ is larger for the fourth glomerulus" | Olsen2010 | — | full-text |
| cause of saturation | "The saturating form of this function reflects the combined effects of short-term depression at ORN-PN synapses and the relative refractory period of PNs" | Olsen2010 | — | full-text |
| exponent choice | "The mean squared error had a minimum for an exponent of 1.5 and 1.6 for glomeruli DL5 and VM7" | Olsen2010 Methods | 1.5 / 1.6 | full-text |
| input-gain normalization (Eq. 2) | "PN=Rmax(ORN1.5ORN1.5+s1.5+σ1.5)" | Olsen2010 | — | full-text |
| response-gain alternative (Eq. 3) worse | "the input gain model generated better fits than the response gain model" | Olsen2010 | — | full-text |
| s linear in total ORN activity (Eq. 4) | "s=m⋅LFP … where the slope m represents the sensitivity of each glomerulus to lateral inhibition" | Olsen2010 | m = 10.63 (VM7), 4.19 (DL5) | full-text |
| LFP from ORN rates (Eq. 5/6) | "LFP=(∑i=124ri)/190mV⋅sec2/spikes" ; "s=m⋅(∑i=124ri)/190" | Olsen2010 | divisor 190; sum over 24 Hallem–Carlson ORN types | full-text |
| population simulation constants | "we used the following parameters for all glomeruli: Rmax = 165 spikes/sec and σ = 12 spikes/sec … The constant m in Equation (6) was set to 10.63 for all glomeruli" | Olsen2010 Methods | Rmax 165, σ 12, m 10.63 (response-gain version m = 0.164) | full-text |
| negative ORN → PN zero | "if the presynaptic ORN odor response was a negative number … then the PN response was set to zero" | Olsen2010 Methods | — | full-text |
| effect | "It also decreases the magnitudes of the strongest population responses while leaving the weaker responses relatively unaffected" | Olsen2010 | — | full-text |
| lateral inhibition scales with total ORN input | "The strength of this inhibitory signal scales with total feedforward input to the entire antennal lobe" | OlsenWilson2008 abstract | lateral input vs total ORN spikes r² = 0.73 (n = 14 odors) | full-text |
| presynaptic, GABAA + GABAB on ORN terminals | "our results imply this is mediated by both GABAA and GABAB receptors on the same nerve terminal" | OlsenWilson2008 | CGP54626 50 µM, picrotoxin 5 µM | full-text |
| GABAB = late phase | "GABAA receptors were required for the a brief early phase of inhibition after odor onset (Fig. 5), while GABAB receptors were required for the long, late phase" | OlsenWilson2008 Discussion | durations not stated in text | full-text |
| purpose | "This should prevent a stimulus from saturating the dynamic range of many PN types simultaneously" | OlsenWilson2008 | — | full-text |
| ORN→PN synapse very strong | "Averaged across PNs, uEPSP amplitude was 6.19 ± 0.45 mV (n = 23)" ; "Mean values are N = 51.4 ± 7.8, q = 1.05 ± 0.11 pA, and p = 0.79 ± 0.02" | Kazama2008 | uEPSC 29.0 ± 2.6 pA at 0.033 Hz | full-text |
| equalized across glomeruli | "unitary synaptic potentials are constant across glomeruli, although unitary synaptic currents are larger in large glomeruli" | Kazama2008 | — | full-text |
| STD magnitude | "At frequencies mimicking the basal firing rate of a typical ORN (7 Hz), synaptic responses depress by about 40% but remain relatively strong" ; "We observed strong depression at all frequencies above about 50 spikes/s" | Kazama2008 Discussion | ~40 % at 7 Hz; strong >50 Hz | full-text |
| STD mainly presynaptic | "this result indicates a mainly presynaptic origin for synaptic depression at this stimulus frequency" | Kazama2008 | 1/CV² vs depression r = 0.79 | full-text |
| PNs not intrinsically adapting | "final firing rates were 104.2% of the initial rate, n = 8 cells" | Kazama2008 Fig. 8C | — | full-text |
| recovery time constant | — | Kazama2008 | **not stated** in main text | full-text |
| STD model and recovery τ | "the amplitude of the unitary postsynaptic conductance decrements by a factor f after each spike, and recovers with a time constant τ between spikes" ; "f = 0.78 and τ = 893 ms" | Nagel2015 | single-component fit: f 0.78, τ 893 ms | full-text |
| two nicotinic components | "For the IMI-resistant component, these parameters were f = 0.77, and τ = 1006 ms, whereas for the curare-resistant component, these parameters were f = 0.91, τ = 629 ms" | Nagel2015 | fast f 0.77 / τ 1006 ms; slow f 0.91 / τ 629 ms | full-text |
| rate-based model constants | "For the fast component, we used r = 0.23 spike−1, τA = 1006 ms, k = 20 nS/spike, and τg = 9.3 ms" ; "Fitted parameters for the slow component were r = 0.0073 spike−1, τA = 33247 ms, k = 1.8 nS/spike, and τg = 80 ms" | Nagel2015 Methods | unitary conductance 0.28 nS (0.22 fast + 0.06 slow); PN Rm 800 MΩ, τm 5 ms, Eleak −70, Esyn −10 mV | full-text |
| LN presynaptic inhibition is slow | "The time course of inhibition could be fit with an alpha function with a time constant of about 25 ms" | Nagel2015 | 25 ms alpha | full-text |
| presynaptic inhibition reduces depression | "because inhibition is presynaptic, it decreases the rate of synaptic depression, thereby preserving synaptic resources" | Nagel2015 | — | full-text |
| Liu 2021 claim: balanced STF+STD at ORN→PN | "we show that a strong STP balanced between short-term facilitation (STF) and short-term depression (STD) is responsible for the observed nonlinear divisive normalization" | Liu2021 abstract | model-fitted (not measured): DL5 τD 368 ms, τF 339 ms, U 0.31; VM7 τD 160 ms, τF 150 ms, U 0.24; τE 50 ms; ωEE 160/105 nS; ρ 1.9/2.5 ms | full-text |
| Liu 2021: without STP the PN is linear | "when we set τ D = τ F = 0, the PN response curves are roughly linear to ORN input within the range of the experimental data" | Liu2021 | Eq. 13 R*=τE·ωEE·U·R/(1+A·ΣRj): "linear" divisive normalization, γ = 1 | full-text |
| Liu 2021: basis for STP claim is citation + fit | "From previous studies (Kazama and Wilson, 2008 ; Martelli and Fiala, 2019 ), the synapses between ORNs and PNs in AL exhibit strong short-term plasticity (STP)" ; model eqs. "(4) dx/dt = (1 − x)/τD − x u+ p R, (5) du−/dt = −u−/τF + U(1 − u−) p R" | Liu2021 | Tsodyks–Markram mean-field; PI p = 1/(1+ρ R_LN) | full-text |
| Liu 2021: fitted STD/STF ratio | "r ≡ S D / S F , remains roughly the same for DL5 ( r ≈ 1.08) and VM7 ( r ≈ 1.07)" | Liu2021 | — | full-text |

### Not found / uncertain (Q3)
- Kazama & Wilson 2008 main text gives no recovery time constant (only Supplemental Procedures, not opened) and reports only depression; "facilitation" occurs only in its reference list. So Liu 2021's STF at ORN→PN is a model inference (fitted to Olsen 2010 data), not a measured phenomenon in the cited Kazama paper. Martelli & Fiala 2019 (also cited for STP) was not opened.
- Olsen 2010 gives no "K" parameter; the equivalents are σ (semi-saturation), s = m·LFP, and the 1/190 LFP scaling. Units for Eq. 5 are garbled in the HTML.
- Olsen & Wilson 2008 gives no fitted gain-control constants and no GABAB time constant in text.
- Nagel 2015's slow-component τA = 33,247 ms was fitted to disinhibited odor responses (the authors say curare fits depressed too fast), so it is a model fit, not a direct measurement.

---

## Q4. KC excitability, thresholds, input-number compensation

### Table 4

| claim | verbatim quote | source | numbers/constants | access |
|---|---|---|---|---|
| publication venue | (Europe PMC record) | Abdel2021 | PNAS 2021, doi 10.1073/pnas.2102158118; bioRxiv 10.1101/2021.02.03.429444 | full-text |
| variability hurts, compensation rescues | "memory performance is rescued while maintaining realistic variability if parameters compensate for each other to equalize KC average activity" | Abdel2021 | coding level 0.1 | full-text |
| parameters varied | "N (number of PN inputs per KC), w (strength of each PN–KC connection), and θ (KC spiking threshold)" ; θ = "spiking threshold minus resting potential; mV" from Turner 2008; N from Caron 2013 | Abdel2021 | w log-normal, N and θ Gaussian fits (values figure-only) | full-text |
| activity-independent rule | "We simulated these correlations ( w ∝ θ ; w ∝ 1 / N ) constrained by experimental data" | Abdel2021 | — | full-text |
| homeostatic rules | "Overly active KCs weaken excitatory input weights ( w ji ; A 2), strengthen inhibitory input weights ( α j ; A 3), or raise spiking thresholds" ; "desired average activity level across all odors, A 0 (with a tolerance of ± 6 % )" | Abdel2021 | ±6 % | full-text |
| connectome: more PN inputs → fewer synapses each | "on average, there were ≈ 6 − 15 % fewer input synapses per PN–KC connection ( w ¯ ) for each additional PN per KC ( N ) (compare with … [–22%] and … [–18%] model parameters" | Abdel2021 | −6 to −15 % per extra PN (hemibrain); model −22 %/−18 % | full-text |
| total PN→KC synapses sublinear in N | "the number of total PN–KC synapses per KC increased only sublinearly relative to the number of PN inputs per KC" | Abdel2021 | — | full-text |
| more inputs → farther from spike zone (αβ-c, γ-main) | "the more PN inputs a KC has, the farther away the input synapses are from the putative spike initiation zone" | Abdel2021 | — | full-text |
| inhibition tracks excitation per KC | "the more total PN–KC synapses there were per KC, the more calyx APL–KC synapses there were" | Abdel2021 | r values figure-only (Fig. 6K) | full-text |
| connectome thresholding | "We ignored PN–KC connections with two or fewer synapses" | Abdel2021 | ≥3 synapses | full-text |
| APL cannot equalize KCs alone | "tuning inhibitory weights cannot compensate on its own for variability in other KC parameters" ; "reduce the fraction of active KCs by half (from 20 to 10%" | Abdel2021 | coding level 0.2 → 0.1 with inhibition; needs 99 % no-inhibition coding level to avoid negative weights | full-text |
| threshold tuning alone needs wider θ than measured | "This larger variance of thresholds suggests that natural variation of θ is too small, on its own, to equalize KC activity" | Abdel2021 | — | full-text |
| model APL simplification | "APL's activity was the sum of all postsynaptic excitation of all KCs (without the KCs' threshold applied)" | Abdel2021 | global inhibition, weight α_j | full-text |
| homeostasis to excess APL inhibition (4 d) | "prolonged (4-d) artificial activation of the inhibitory APL causes increased Kenyon cell odor responses" | ApoLin2020 | 88–96 h at 31 °C | full-text |
| little compensation for loss of APL | "Kenyon cells show little, if any, homeostatic compensation for prolonged lack of inhibition from APL" | ApoLin2020 | acute 16–24 h vs constitutive TNT | full-text |
| time course | "only at 4 d did we consistently observe significantly higher KC odor responses" ; "adaptation does not last more than 1 to 2 d after excess inhibition from APL stops" | ApoLin2020 | 4 d on / 1–2 d off | full-text |
| mechanism, subtype-specific | "while adaptation in γ KCs can be explained by decreased APL odor responses, adaptation in αβ KCs requires an additional mechanism" | ApoLin2020 | — | full-text |
| KCs need several active claws | "there was a prominent increase in the proportion of spiking responses between 3 and 4 contacted claws" ; "KCs have 7 claws on average … strongly activating more than half of the dendritic inputs is required to drive a KC to spike" | Gruntman2013 | 3→4 claws; mean 7 claws | full-text |
| spiking is rare | "Only 2 of 39 KCs climbed above spike threshold" | Gruntman2013 | 2/39 | full-text |
| the 2 spiking cells had many claws | "These responses were found only in cells with three or five connected claws" | Gruntman2013 | 3 or 5 claws | full-text |
| requirement not absolute | "KCs have a strong but not absolute requirement for activation of multiple claws" ; spiking cells with "1, 2, 3, 4 and 6 claws (n=2 KCs, n=3, n=1, n=6 and n=1" | Gruntman2013 | n = 191 recorded; spike count vs claws R² = 0.02 | full-text |
| imaging: six-seven claws always drive soma | "odors that activated six or seven claws invariably elicited a somatic response" | Gruntman2013 | — | full-text |
| linear/sublinear summation, passive dendrites | "claws interact linearly when small numbers are coactive, and sub-linearly when larger numbers are stimulated" | Gruntman2013 | — | full-text |
| linear PN summation (Inada) | "responses were always combined linearly or slightly sublinearly in all KC types" ; "linear fit for all KCs (slope = 0.85, R2 = 0.81" | Inada2017 | slopes 0.89/0.75/0.86 (α/β, α′/β′, γ) | full-text |
| intrinsic threshold differs by type | "α′/β′ KCs had a lower firing threshold compared to the other two cell types" ; input resistance "not significantly different between cell types" | Inada2017 | threshold values figure-only | full-text |

### Not found / uncertain (Q4)
- Abdelrahman 2021 numeric distributions of θ, w, N (means/SDs) and the Pearson r values of Fig. 6K are in figures/SI, not in text.
- Apostolopoulou & Lin 2020 effect sizes (ΔF/F) are figure-only; no intrinsic-threshold change was measured electrophysiologically.
- No source normalizes KC threshold by total PN input weight; the closest biological statements are the inverse N–w correlation and APL–PN synapse co-scaling (Abdel2021).

---

## Q5. Glomerular bias of PN→KC input

### Table 5

| claim | verbatim quote | source | numbers/constants | access |
|---|---|---|---|---|
| non-uniform glomerular input, most frequent | "Inputs from the DA1 and DC3 glomeruli are most frequent, with each accounting for 5.1% of the total connections" | Caron2013 | 665 connections, 200 KCs, 53 inputs (51 glomeruli + 2 thermo) | full-text |
| cause = bouton number/size | "The non-uniform distribution reflects the fact that the size and number of calycal boutons formed by PNs varies across glomeruli" | Caron2013 | — | full-text |
| otherwise random | "each KC receives input from a combination of glomeruli randomly chosen from the non-uniform distribution of glomerular projections to the MB" | Caron2013 | only 11/200 KCs had 2 inputs from same glomerulus | full-text |
| per-type PN count and boutons vary | "glomeruli responsive to sex pheromone (DA1 and VA1v …) and … geosmin (DA2 …) had many more PNs per glomerulus than other types … but made relatively few boutons per PN" ; DC4 "both PN number and boutons per PN were low, resulting in small net input to the MB calyx" | Zheng2022 | per-type values figure-only (Fig. 2A–C) | full-text |
| food PNs over-sampled, per glomerulus | "Food-responsive PNs provided output to more claws than non-food PNs on both a per-bouton (H, mean ± s.d., 15.66 ± 3.08 vs. 10.53 ± 7.3 …) and per-glomerulus (J, mean ± s.d., 162.95 ± 60.74 vs. 100.06 ± 56.17" | Zheng2022 | claws/glomerulus 163 ± 61 vs 100 ± 56 | full-text |
| over-convergence vs random-bouton null | "The observed number of claws receiving input from core community PNs (1,916; red dot) greatly exceeds the random bouton null model prediction (… 1,421.7 ± 35.7; z-score, 13.8)" | Zheng2022 | ≈1.35× null | full-text |
| claws per bouton | "Core community PNs have more claws per bouton than other PNs (mean ± s.d., 17.4 ± 9.3 vs. 11.5 ± 7.4" | Zheng2022 | — | full-text |
| mostly explained by geometry | "the overconvergent PN core community disappeared (Figure 5E–F), indicating that this model largely captures the observed network structure" (local random null) | Zheng2022 | 1,916 observed vs 1,890.6 ± 22.5 local-random | full-text |
| stereotyped across animals | "the fraction of output per PN type is highly correlated across the two datasets (r2=0.83" (FAFB vs hemibrain) | Zheng2022 Fig. 1 legend | r² = 0.83 | full-text |
| claws per KC | "The average number of claws per KC, 5.2 ± 1.6" | Zheng2022 Methods | 7,102 claws / 1,356 KCs mapped | full-text |
| hemibrain: KC-subtype-specific over-representation | "our analysis revealed subtype-specific biases in PN sampling by KCs, including an overrepresentation of specific glomeruli by α/β and α′/β′ KCs (Figure 13B)" | Li2020 | per-glomerulus probabilities figure-only | full-text |
| hemibrain anatomy | "A small number (generally 3 – 4) of PNs from each of the 51 olfactory glomeruli innervate the MB" ; uPNs "63.6% of total input to the KCs" | Li2020 | 3–4 PNs/glomerulus; 129 uPNs | full-text |
| upstream stages equalize glomeruli | "unitary synaptic potentials are constant across glomeruli" (Kazama2008); "Rmax and σ are essentially the same for all glomeruli" (Olsen2010); "These transformations distribute activity more uniformly among the glomerular channels in the PN layer than the ORNs" (Gruntman2013 Intro) | Kazama2008; Olsen2010; Gruntman2013 | PN Rmax ≈ 144–170 spikes/s | full-text |
| per-KC compensation of input number | "≈ 6 − 15 % fewer input synapses per PN–KC connection … for each additional PN per KC" | Abdel2021 | — | full-text |

Derived (my arithmetic, not a source statement): Caron 2013's 5.1 % for the top glomeruli vs 1/53 ≈ 1.9 % for uniform sampling is ≈2.7× over-representation; Zheng 2022's food vs non-food per-glomerulus claw means differ ≈1.6×, with SDs of 56–61 claws around means of 100–163.

### Not found / uncertain (Q5)
- No source states in text the full range (max/min) of per-glomerulus KC input (claws or synapses); extremes are figure-only (Zheng 2022 Fig. 2C/3I, Li 2020 Fig. 3 pie and Fig. 13B, Caron 2013 Fig. 3 / Supp. Fig. 1). I therefore cannot verify how large the biological max/min ratio is.
- Every stated magnitude is a few-fold effect (≈1.35× vs null, ≈1.6× food vs non-food, ≈2.7× top vs uniform). Nothing I read supports KC drive differing by ~10³ (e.g. 2867×) across glomeruli at equal ORN drive. In addition, the biology has ORN→PN equalization (uEPSP constant across glomeruli; common Rmax/σ; divisive normalization) that the model lacks. Whether per-glomerulus synapse totals in a connectome could reach a large ratio for the rarest glomeruli is not stated in these papers. Inference, not a sourced claim: a ~2867× disparity is more likely a model artifact (e.g. rare glomeruli with near-zero KC synapses combined with unnormalized PN drive and per-KC threshold normalization) than an expected biological property.
