# LHPV3c1: identity and neurotransmitter (literature and data check)

Checked 2026-09-17. Every quote below was matched character for character (after whitespace normalisation) against a local copy of the source, using `src/lhp/vq.py` + `src/lhp/quotes.json` (67/67 found). Local copies are in `m0d-lit/src/lhp/`.

What the access labels mean:
- **full text**: I read the whole article (eLife HTML, or PMC OAI XML).
- **data file**: I downloaded the published supplementary or data table and read the rows myself.
- **abstract-only**: I could only get the abstract.
- **not accessible**: I could not open it (login or paywall).

## 0. Bottom line

| Question | Answer | Strength |
|---|---|---|
| Identity | **LHPV3c1 = MB-CP2** ("Mushroom Body Calyx Pedunculus #2"). Zheng et al. 2018 (FAFB) named it, and the hemibrain gave it the systematic lateral horn name LHPV3c1. There is one cell per hemisphere. | Strong. Li et al. 2020 gives the two names side by side with the same hemibrain body ID. The FBbt ontology lists them as exact synonyms. |
| Same as MB-C1? | **No.** MB-C1 is a separate GABAergic type. It has 2 cells per hemisphere (hemibrain R: 2; FlyWire: 2L+2R; MaleCNS: 2L+3R), a different hemilineage and cell body fibre, and separate body IDs in every dataset. | Strong |
| Same as MB-CP1? | **No.** MB-CP1 is MBON22 ("MBON-calyx"), an atypical MB output neuron. | Strong |
| Transmitter | **Acetylcholine (predicted).** Two EM volumes give independent machine-learning predictions with high confidence: hemibrain 0.897, FlyWire 0.835/0.894. LHPV3c1 was not in the training ground truth for either. MaleCNS agrees (0.917/0.952), but its model was trained with LHPV3c1 labelled as ACh. | Moderate to strong for the prediction. **Weak for direct experimental evidence:** databases cite an ACh label as "Dolan et al., 2019 (immuno)", but I could not trace that label to Dolan 2019. FBbt says it was predicted from hemilineage (see §2b). |
| KC → LHPV3c1 → KC loop | The anatomy is documented. Zheng 2018 reports that KC axons in the pedunculus synapse onto MB-CP2, which sends boutons back to KC claws, and calls this "recurrent feedback". If the neuron is cholinergic, the loop is excitatory in sign, but no physiology exists. | Anatomy: strong. Sign: inferred |

---

## 1. Identity: what LHPV3c1 is in the literature

### 1a. Naming and synonym evidence

| Claim | Verbatim quote | Source | Access |
|---|---|---|---|
| Zheng 2018 discovered and named MB-CP2 in FAFB, using Tanaka 2008's naming convention | "Seventeen boutons (3%) arose from a previously unknown neuron that we named" … "Mushroom Body Calyx Pedunculus #2," … "per the naming convention of Tanaka et al. (2008)" | Zheng et al. 2018 *Cell* 174:730, DOI 10.1016/j.cell.2018.06.019, PMC6063995 | full text |
| MB-CP2 is a new type providing input to KC claws | "A Previously Unknown Cell Type, MB-CP2, Provides Input to KC Claws" | same | full text |
| One cell per hemisphere | "A second MB-CP2 neuron was located in the left hemisphere." | same | full text |
| **Hemibrain synonym: MB-CP2 = LHPV3c1**, body ID 479935033 | "Another LVIN, MB-CP2 (LHPV3c1) (479935033), provides 12.6% of the input to KCs in the dACA; however, only a small percentage of its inputs are visual" | Li et al. 2020 *eLife* 9:e62576, DOI 10.7554/eLife.62576 (https://elifesciences.org/articles/62576) | full text |
| Same synonym again (Figure 11, figure supplement 2E legend) | "MB-CP2 (LHPV3c1) also provides 1% of input to KCs in the CA" | Li et al. 2020, figures page https://elifesciences.org/articles/62576/figures | full text |
| Ontology: exact synonyms | Label "mushroom body calyx-pedunculus arborizing neuron 2"; synonyms "LHPV3c1", "MB-CP2" | FBbt:00048241 via OLS4 API (https://www.ebi.ac.uk/ols4/api/ontologies/fbbt/terms?obo_id=FBbt:00048241) | data file (JSON) |
| What the LH name means | "LHPV1c1-12a1 (Lateral Horn Posterior Ventral cell cluster [cell cluster ID][anatomy group ID][type ID])" | Scheffer et al. 2020 *eLife* 9:e57443, DOI 10.7554/eLife.57443 | full text |
| Where the hemibrain LH names come from | "We named these cells by extending the LHN naming scheme from Frechter et al., 2019, except for cell types with more prominent names already in use in the literature." | Schlegel et al. 2021 *eLife* 10:e66018, DOI 10.7554/eLife.66018 | full text |
| Origin of the name "PV3c1": a single FlyCircuit light-microscopy neuron in Frechter 2019 | Row: `"Cha-F-000350","PV3c1","PV3c","PV3"," ","ON","FlyCircuit"` | Frechter et al. 2019 *eLife* 8:e44590, file `elife-44590-supp3-v2.csv` (skeleton metadata with cell type annotations; the page captions for files 3 and 4 appear swapped relative to the file names) | data file |
| Same type in all three connectomes | FlyWire: 2 neurons, `cell_type`=`hemibrain_type`=LHPV3c1, left and right, `ito_lee_hemilineage` putative_primary. Hemibrain: `479935033,LHPV3c1_R,LHPV3c1,…,PVL10,right`. MaleCNS: bodyIds 11688 (LHPV3c1_L) and 11903 (LHPV3c1_R), `flywireType`=`hemibrainType`=LHPV3c1, `synonyms` empty | FlyWire `Supplemental_file1_neuron_annotations.tsv` and `Supplemental_file5_hemibrain_meta.csv` (github.com/flyconnectome/flywire_annotations); MaleCNS `body-annotations-male-cns-v1.0-minconf-0.5.feather` (male-cns.janelia.org/download; SHA1 identical to the repo's `data/raw` copy) | data file |

Checked with no hit for LHPV3c1 or MB-CP2: Bates et al. 2020 *Curr Biol* (PMC7443706; mentions MB-C1 only), Schlegel et al. 2024 *Nature* main text (PMC11446831), Dorkenwald et al. 2024 *Nature* main text (PMC11446842), Dolan et al. 2019 *eLife* main text and supplementary files 1–2, Frechter 2019 main text, Eckstein 2024 main text, and Ganguly et al. 2024 *Nat Commun* (PMC11228034). A WebSearch summary claimed Ganguly mentions LHPV3c1; the full text does not.

### 1b. Connectivity reported for MB-CP2/LHPV3c1, compared with the model

| Claim | Verbatim quote | Source | Access | Matches model? |
|---|---|---|---|---|
| Receives input from γ KCs, including visual γd KCs, in the pedunculus | "MB-CP2 receives input from γ KCs"; "γd KCs, a subtype originating in the ventral accessory calyx known to receive visual inputs" | Zheng 2018 | full text | Yes (model: KCg-m, KCg-d input) |
| Outputs onto KCs in the main calyx | "MB-CP2 boutons are presynaptic to all five known olfactory KC subtypes (γ, αβc, αβs, α′β′m, α′β′ap) in MB main calyx" | Zheng 2018 (Video S5 legend) | full text | Yes (model: KCg-m, KCab-c/s/m, KCa'b'-m) |
| Outputs onto αβp KCs in the dorsal accessory calyx | "In the dorsal accessory calyx (dAC), MB-CP2 boutons also provide input to αβp KCs (data not shown)." | Zheng 2018 | full text | Yes (model: KCab-p) |
| Strongest single input to the dACA | "One LVIN, MB-CP2, which has been suggested to integrate multi-sensory inputs (Zheng et al., 2018), is the single strongest dACA input neuron" | Li 2020 | full text | consistent |
| Recurrent loop, described explicitly | "MB-CP2 neurons likely relay multimodal, non-olfactory input to KCs, and provide recurrent feedback from KC axons in the MB pedunculus to KC dendrites in the MB main calyx" | Zheng 2018 | full text | Yes (the model's KC→LHPV3c1→KC loop) |
| Its output synapses are PN-like boutons | "at the center of a canonical microglomerulus" | Zheng 2018 (Fig. 6D legend) | full text | n/a |
| Visual input is only a minor share | "only a small percentage of its inputs are visual" | Li 2020 | full text | Yes (model: MeVP38 among several inputs) |
| Relays input from the SEZ | "also seems to relay input from the subesophageal zone (SEZ)" | Li 2020 | full text | not checked in model |
| Ontology summary | "It is presynaptic to a range of Kenyon cell (KC) subtypes, including alpha/beta posterior KCs" | FBbt:00048241 | data file | yes |

**Conclusion for Q1:** the model's LHPV3c1 is MB-CP2. The MaleCNS/FlyWire/hemibrain type label says so directly, and its reported connectivity (KC input in the pedunculus, bouton output to many KC subtypes, especially α/βp) matches Zheng 2018 and Li 2020.

### 1c. Is it MB-C1 or MB-CP1? No.

| Claim | Verbatim quote / data | Source | Access |
|---|---|---|---|
| MB-C1 is a separate, GABAergic calyx/LH type | "Two GABAergic MB-C1 neurons (shown in different colors) innervate the LH and CA" | Li 2020 Fig. 3, figure supplement 1H | full text |
| MB-C1 appears in the same hemibrain analysis as a separate KC input from MB-CP2 | "from APL (9.5%; see Figure 3—figure supplement 1A) and from MB-C1 (4.6%" … and, in the same legend, "2, MB-CP2 (1.0%)" | Li 2020 Fig. 9 legend | full text |
| MB-C1 in FAFB: putative inhibitory; Zheng traced 2 MB-C1 **and** 2 MB-CP2 as distinct neurons | "MB-C1, a putative inhibitory interneuron that innervates the MB calyx and LH"; "Two MB-C1 neurons were found in the EM-based survey of KC postsynaptic targets, in contrast to the single neuron reported by Tanaka et al. (2008)" | Zheng 2018 | full text |
| MB-C1 is GABAergic (Aso 2014) | "one type of GABAergic neuron (MB-C1)"; Table 1 row lists GABA / MB-C1 / >2 / MB380B / Tanaka 2008 | Aso et al. 2014 *eLife* 3:e04577, DOI 10.7554/eLife.04577 | full text |
| MB-C1 is GABA by Dolan's immunostaining | Supp. file 1 row: `MB-C1  Output  L2449  L1900  GABA`. Method: "we performed immunohistochemistry for each of the three main neurotransmitters in the fly brain (glutamate, acetylcholine and GABA)" | Dolan et al. 2019 *eLife* 8:e43079, DOI 10.7554/eLife.43079, `elife-43079-supp1-v1.xlsx` | full text + data file |
| MB-C1 was a GABA training example in Eckstein 2024 | Data S1: `MB-C1,GABA,Dolan et al. 2019,immuno` | Eckstein et al. 2024 *Cell* 187:2574, DOI 10.1016/j.cell.2024.03.016, PMC11106717, mmc2.csv (via Europe PMC supplementaryFiles) | data file |
| Different lineage and cell body fibre | Hemibrain meta: LHPV3c1 cbf **PVL10**, hemilineage putative_primary; MB-C1 cbf **PVL03**, hemilineage VPNp&v1_posterior | FlyWire `Supplemental_file5_hemibrain_meta.csv` | data file |
| MB-CP1 is MBON-calyx (MBON22) | "one neuron (MBON-calyx, or MB-CP1) extends dendrites in the main calyx, the posterior part of the pedunculus and the ventral part of the lateral horn, and thus we classify this as an atypical MB output neuron"; Table 1 row lists MBON-calyx / MBON-22 / … / MB-CP1 | Aso 2014 | full text |
| MB-CP1 in FAFB | "MB-CP1, a MB output neuron (MBON) with a dendritic arbor innervating the MB calyx and pedunculus" | Zheng 2018 | full text |
| Tanaka 2008 original description of MB-C1 or MB-CP1 | Abstract says only "we resolved 17 other types of MBENs that arborize in the calyx, lobes, and pedunculus". It does not mention MB-C1 or GABA. | Tanaka, Tanimoto, Ito 2008 *J Comp Neurol* 508:711, DOI 10.1002/cne.21692 (PubMed 18395827; OpenAlex `oa_status: closed`; Wiley returned 403) | **abstract-only** |

---

## 2. Neurotransmitter evidence

### 2a. Machine-learning predictions from EM

| Dataset / source | Neuron(s) | Prediction and confidence (verbatim table values) | In training ground truth? | Access |
|---|---|---|---|---|
| **Eckstein 2024, Data S3 (hemibrain v1.2.1)** | 479935033 | Row `479935033,LHPV3c1,1221,FALSE,acetylcholine,0.897,acetylcholine,0.892,0.9`. Columns: `bodyid, cell_type, pre, cropped, conf_nt, conf_nt_p, top_nt, top_nt_p, acetylcholine` (0.9 of synapse votes). Other transmitters 0. | `in_ground_truth` = FALSE | data file (mmc4.csv) |
| **Eckstein 2024, Data S4 (FAFB-FlyWire)** | root_783 720575940615438495 (L), 720575940630914245 (R) | L: conf_nt acetylcholine, conf_nt_p **0.835**, top_nt_p 0.964, votes ACh 0.8 / DA 0.1. R: conf_nt_p **0.894**, top_nt_p 0.965, votes ACh 0.9 / DA 0.1. `known_nt` acetylcholine, `known_nt_source` "Dolan et al., 2019" | `in_ground_truth` = FALSE for both | data file (mmc5.csv) |
| **Eckstein 2024, Data S7 (cell-type level)** | LHPV3c1 | `LHPV3c1,acetylcholine,0.98,acetylcholine,"Dolan et al., 2019"`. Hemibrain type-level: acetylcholine, 0.94 | LHPV3c1 absent from Data S1 (training cell types) and Data S2 (training neurons) | data file (mmc8/mmc2/mmc3) |
| How to read those scores (Eckstein) | "we suggest that if users wish to be conservative they could use a stringent neuron-level transmitter prediction confidence score threshold of ∼ 0.62 and ∼ 0.53 for FAFB-FlyWire and HemiBrain neurons respectively". LHPV3c1 exceeds both. | — | full text |
| Eckstein accuracy on known cholinergic types | "We predicted most known cholinergic cell types correctly (FAFB-FlyWire, 91%; HemiBrain, 91%)" | — | full text |
| **FlyWire annotations v3 (Schlegel 2024 / Berg 2025 update)** | same 2 neurons | `top_nt` acetylcholine; `top_nt_conf` 0.9643581532030119 (L), 0.9651121782040202 (R); `known_nt` acetylcholine; `known_nt_source` "Dolan et al., 2019 (immuno)". README: "`top_nt_conf` is the average confidence for the top neurotransmitter" | — | data file |
| **MaleCNS v1.0** | 11688 (L), 11903 (R) | `predicted_nt` acetylcholine, `predicted_nt_confidence` 0.916799 / 0.951601; `celltype_predicted_nt_confidence` 0.932012; `consensus_nt` acetylcholine; **`ground_truth` acetylcholine** | **Yes, labelled.** The MaleCNS ground-truth CSV has `putative_primary,,LHPV3c1,,"Dolan et al., 2019",immuno,5,1,-1,-1`, above the stated filter: "We then filtered the ground truth to exclude entries with an evidence confidence rating below 3". MaleCNS is therefore **not independent** of that label. | data file (body-neurotransmitters feather; funkelab/drosophila_neurotransmitters `gt_sources/male_cns/202509-male_cns_gt_data.csv`); preprint full text (Berg et al. 2025 bioRxiv, DOI 10.1101/2025.10.09.680999, PMC12636603) |
| Shuai et al. 2025 (secondary use of Eckstein) | hemibrain LHPV3c1 | Supp. file 3 row ends `acetylcholine 0.939393939393939`. Caption: "Neurotransmitter (NT) prediction data were from Eckstein et al., 2023, and the fraction of synapses predicted for the neurotransmitter was pooled from all cells of the cell type." | — | data file; Shuai et al. 2025 *eLife* 13:RP94168, DOI 10.7554/eLife.94168 |
| FlyWire Codex cell page | — | Page shows only a sign-in landing page | — | **not accessible (Google login)** |
| neuPrint (hemibrain / MaleCNS) web UI | — | Not opened (needs login). I used the public MaleCNS bulk tables instead. | — | **not accessible** |

**Known caveats for ML predictions near the MB** (full text, Eckstein 2024):
- "Kenyon cells had been mispredicted for dopamine in both HemiBrain and FAFB-FlyWire rather than acetylcholine"
- On the possible cause: "features from proximal presynapses may have skewed the result"

My inference, not a published statement: part of LHPV3c1's output is in the calyx, where boutons sit next to cholinergic PN boutons. Contamination could in principle inflate an ACh call. Two things argue against that being the whole story. LHPV3c1 also has outputs in the LH and SLP. And the call agrees across hemibrain, both FlyWire hemispheres and MaleCNS, with near-zero GABA and glutamate votes.

### 2b. Experimental (immunostaining / genetic) evidence: tracing the "Dolan et al., 2019 (immuno)" label

| Step | Verbatim quote / data | Source | Access |
|---|---|---|---|
| Databases cite Dolan 2019 immunostaining for LHPV3c1 = ACh | FlyWire: `acetylcholine  Dolan et al., 2019 (immuno)`. MaleCNS GT: `"Dolan et al., 2019",immuno,5,1,-1,-1` | as above | data file |
| **But Dolan 2019 lists no PV3c1.** Its supplementary file 1 (66 cell types with a Neurotransmitter column) has only one PV3 type. Supplementary file 2 (lines) also has only PV3f1 lines. | Row `PV3f1  Output  L578  L206  Acetylcholine`. There is no "PV3c" string in the main text, the figures page, or supp. files 1–2. | Dolan et al. 2019, `elife-43079-supp1-v1.xlsx`, `elife-43079-supp2-v1.xlsx` | full text + data file |
| **The ontology says the ACh label is a hemilineage-based inference** | "Neurotransmitter predicted based on hemilineage (Schlegel et al., 2021)." and "it fasciculates with the PV3 primary neurite tract and it is cholinergic (Schlegel et al., 2021)" | FBbt:00048241 (OLS4) | data file |
| Schlegel 2021's method is extrapolation within a hemilineage from Dolan's immunostaining | "assigned all members of a given hemilineage the same ’transmitter identity’ if we knew that at least one member of that hemilineage to express acetylcholine, GABA or glutamate based on immunohistochemical work (Dolan et al., 2019)"; "This is a useful proxy that gives an impression of fast-acting neurotransmitter expression diversity throughout the pool of TOONs, but it is far from definitive." | Schlegel 2021 | full text |
| The same "Dolan 2019 immuno, confidence 5" label is applied to other LHPV3 types, none of which are in Dolan's table | e.g. `VPNp1_medial,,LHPV3a1,,"Dolan et al., 2019",immuno,5,1,-1,-1` (also LHPV3a2, LHPV3b1_b) | MaleCNS GT CSV | data file |
| Supporting genetic hint: the light-microscopy neuron that defined PV3c1 came from a ChAT-GAL4 MARCM clone | VFB record for Cha-F-000350 (VFB_00004372; synonym ChaMARCM-F001428_seg001): "[expresses](RO_0002292): [P{ChAT-GAL4.7.4}](FBtp0014830)" | Virtual Fly Brain term-info API (v3-cached.virtualflybrain.org/get_term_info?id=VFB_00004372) | data file |
| No split-GAL4 driver for LHPV3c1 in the Janelia MB driver collection, so no split-line immunostaining is likely | Supp. file 5 row: `LHPV3c1  -` (empty Split-GAL4 column). MB-C1, by contrast, has `MB380B`, `MB380C, SS23817`. | Shuai et al. 2025, `elife-94168-supp5-v1.xlsx` | data file |

**My reading (inference):**
- I found **no direct immunostaining of LHPV3c1/MB-CP2 in any source I could open.**
- The "known ACh" label most likely comes from Schlegel 2021's extrapolation of Dolan 2019's PV3f1 immunostaining (ChAT+) to other PV3-tract neurons. FBbt states that route explicitly.
- A weaker genetic hint is consistent with it: the FlyCircuit exemplar that defined PV3c1 was a ChAT-GAL4 clone. That only holds if the hemibrain type truly matches that light-microscopy neuron. The lhns hemibrain LHN table lists `LM.match` "none" for 479935033, so that match was not documented at neuron level.
- The ACh call therefore rests mainly on EM machine learning (Eckstein hemibrain + FlyWire, independent of the label), with circumstantial lineage and driver support.

### 2c. Transmitter conclusion

**Acetylcholine, i.e. excitatory in a LIF sign convention.** Two separate animals and three volumes agree. The two predictions that did not use the label as training data score 0.835–0.897 (neuron level) and 0.94–0.98 (type level). Nothing I found points to GABA or glutamate: in Eckstein's per-transmitter votes, GABA and glutamate are 0 in both datasets. Direct experimental confirmation was not found.

---

## 3. MB-extrinsic neurons that both receive KC input and send output to KCs

| Neuron | KC → neuron | Neuron → KC | Visual input? | Transmitter (evidence) | Cells | Matches model LHPV3c1 (2 cells, cholinergic, KC in/out, visual in)? |
|---|---|---|---|---|---|---|
| **MB-CP2 = LHPV3c1** | Yes: "MB-CP2 receives input from γ KCs" (Zheng 2018) | Yes: boutons onto 5 olfactory KC subtypes plus αβp (Zheng 2018); 12.6% of dACA and 1% of CA KC input (Li 2020) | Minor ("only a small percentage of its inputs are visual", Li 2020) | ACh (ML predictions; label is lineage-inferred; see §2) | 1/hemisphere | **Yes, this is the neuron** |
| MB-C1 | Yes: "The γ KC from (D, inset) is presynaptic to MB-C1 (pink), APL (green)" (Zheng 2018 Fig. 7F) | Yes: 4.6% of main-CA KC input (Li 2020); 0.7% in lACA | Some. Ganguly 2024 treats it as receiving visual projection neuron input: "aMe12s (top) synapse with LVIN PLP231, with one aMe12 synapsing with MB-C1." | **GABA** (Dolan 2019 immunostaining, supp1; Aso 2014 Table 1; Li 2020 "Two GABAergic MB-C1 neurons"; Eckstein type-level GABA 0.96 FAFB / 0.8 hemibrain; MaleCNS gaba 0.79) | 2/hemisphere (MaleCNS 2L+3R) | No: different cell count, lineage and transmitter |
| APL | Yes | Yes (whole MB; 9.5% of CA KC input, Li 2020) | Indirect | **GABA**: "APL is GABAergic and provides negative feedback important for sparse coding of odor identities" (Li 2020). FlyWire known_nt "Liu et al. 2009 (immuno)"; Eckstein type 0.96 | 1/hemisphere | No |
| DPM | Yes (lobes) | Yes (lobes) | — | Multiple: "The DPM neuron has been proposed to use the neuropeptide amnesiac (Waddell et al., 2000), serotonin (Lee et al., 2011), and GABA (Haynes et al., 2015) as neurotransmitters" (Li 2020). ML mispredicts it: "DPM were not used in our ground truth data and in both datasets are predicted to be dopaminergic." (Eckstein 2024). Ganguly 2024: "the serotonergic neuron DPM forms an excitatory bridge between visual and olfactory Kenyon cells" (a claim about another study; that study was not opened) | 1/hemisphere | No (lobes only; no calyx boutons) |
| MB-CP1 = MBON22 (MBON-calyx) | Yes: "two other γ KCs … presynaptic to APL (green), MB-CP1 (red)" (Zheng 2018 Fig. 7E; quote shortened) | Occasional: some claws receive input from "Lateral Horn neurons (LHNs), interneurons (e.g. APL), or MBONs (e.g. MB-CP1)" (Zheng et al. 2022 *Curr Biol*, DOI 10.1016/j.cub.2022.06.031, PMC9413950) | not reported | ACh by ML only: FlyWire top_nt acetylcholine 0.866/0.873; hemibrain type 0.94; MaleCNS 0.93, ground_truth None. Eckstein FAFB type-level "unknown", no known_nt. | 1/hemisphere | Partially (ACh, KC in), but it is an MBON with dendrites in the calyx, not a bouton-forming KC input |

Only LHPV3c1 itself fits all of: 1 cell per hemisphere, cholinergic, heavy KC input in the calyx and dACA, KC input from γ/γd, and a small visual share. MB-C1 is the only other candidate with KC in, KC out and visual input, but it is GABAergic, has 2–3 cells per side, and has its own type label.

---

## 4. Is the model's positive-feedback sign plausible?

- **Anatomy (published):** KC axons (pedunculus) → MB-CP2 → KC claws is described as "recurrent feedback" (Zheng 2018).
- **KC sign (published):** "Immunohistochemistry and RNA sequencing data have shown that Kenyon cells express the machinery necessary for cholinergic transmission." (Eckstein 2024). So KC→LHPV3c1 is excitatory.
- **LHPV3c1 sign (predicted):** ACh, with high and consistent ML confidence. Its calyx outputs are microglomerular boutons like excitatory PN boutons ("at the center of a canonical microglomerulus", Zheng 2018). That structural point is circumstantial.
- **Inference:** a net excitatory KC→LHPV3c1→KC loop is plausible in sign. Nothing I found gives its gain. There are no physiological recordings or perturbations of MB-CP2. In the fly, APL's GABAergic feedback onto the same KCs runs in parallel (9.5–10.9% of calyx/dACA KC input, Li 2020). If the model runs away, the likelier culprits are weight scaling or missing inhibition, not the sign of LHPV3c1.

---

## 5. Not verified / not accessible (explicit)

1. **Tanaka et al. 2008** (J Comp Neurol 508:711): full text paywalled (Wiley 403, OpenAlex closed). **Abstract-only.** I could not check the original MB-C1 GABA immunostaining statement, nor whether Tanaka 2008 described anything like MB-CP2. All "MB-C1 is GABAergic" quotes above are secondary (Aso 2014, Li 2020, Zheng 2018), plus Dolan 2019's own supplementary table.
2. **Direct immunostaining or genetic transmitter evidence for LHPV3c1/MB-CP2:** **not found.** FlyWire, Eckstein and MaleCNS cite "Dolan et al., 2019 (immuno)", but PV3c1 is absent from Dolan 2019's main text and supp. files 1–2. FBbt says the label is hemilineage-inferred. I did not check the image data behind Dolan 2019 Figure 2 (JPEG panels) or the Janelia split-GAL4 image database.
3. **FlyCircuit Cha-F-000350 → hemibrain LHPV3c1 match:** not verified at neuron level (lhns table `LM.match` = "none"). Whether ChAT-GAL4.7.4 expression in that clone reflects real ChAT expression was not checked.
4. **FlyWire Codex cell-type page** and **neuPrint web pages**: not accessible (login). I used the public bulk tables instead (FlyWire annotations GitHub; MaleCNS feather files).
5. **Eckstein 2024 per-synapse location breakdown** (calyx vs LH/SLP) for LHPV3c1: not available in the supplementary files. The contamination caveat in §2a is untested.
6. **Physiology or behaviour of MB-CP2/LHPV3c1:** none found. A WebSearch for split-GAL4, functional or feedback studies returned nothing, and Shuai 2025 lists no split-GAL4 line.
7. **MaleCNS paper, Cell 2026 version** (ScienceDirect S0092867426009426): not opened. I used the bioRxiv/PMC preprint (PMC12636603) for the neurotransmitter methods. I did not search the preprint for LHPV3c1-specific text beyond a string search (no hits).
8. **The model's specific upstream partners** (MeVP38, M_l2PNm14, LHAV3o1): not checked against the literature. Li 2020 only supports "small visual share" and "SEZ input" in general terms.
9. **The DPM "excitatory bridge" study** cited by Ganguly 2024: not opened; quoted only as Ganguly's statement.

## 6. Sources opened (URL / DOI)

- Zheng et al. 2018 Cell, DOI 10.1016/j.cell.2018.06.019. PMC OAI: `https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:6063995&metadataPrefix=pmc`
- Li et al. 2020 eLife, DOI 10.7554/eLife.62576. https://elifesciences.org/articles/62576 and /figures
- Aso et al. 2014 eLife, DOI 10.7554/eLife.04577. https://elifesciences.org/articles/04577
- Scheffer et al. 2020 eLife, DOI 10.7554/eLife.57443. https://elifesciences.org/articles/57443
- Schlegel et al. 2021 eLife, DOI 10.7554/eLife.66018. https://elifesciences.org/articles/66018
- Dolan et al. 2019 eLife, DOI 10.7554/eLife.43079. https://elifesciences.org/articles/43079, /figures, supp1 and supp2 xlsx
- Frechter et al. 2019 eLife, DOI 10.7554/eLife.44590. https://elifesciences.org/articles/44590/figures, `elife-44590-supp3-v2.csv`
- Eckstein et al. 2024 Cell, DOI 10.1016/j.cell.2024.03.016. PMC11106717 (OAI XML) and `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11106717/supplementaryFiles` (mmc2/3/4/5/8.csv)
- Schlegel et al. 2024 Nature, DOI 10.1038/s41586-024-07686-5 (PMC11446831, main text). Annotations: https://github.com/flyconnectome/flywire_annotations/tree/main/supplemental_files (files 1, 5, README)
- Dorkenwald et al. 2024 Nature, DOI 10.1038/s41586-024-07558-y (PMC11446842, main text)
- Bates et al. 2020 Curr Biol, DOI 10.1016/j.cub.2020.06.042 (PMC7443706)
- Berg et al. 2025 bioRxiv (MaleCNS), DOI 10.1101/2025.10.09.680999 (PMC12636603). Data: https://male-cns.janelia.org/download/ (`body-annotations-…feather`, `body-neurotransmitters-…feather`). GT: https://github.com/funkelab/drosophila_neurotransmitters/blob/main/gt_sources/male_cns/202509-male_cns_gt_data.csv
- Ganguly et al. 2024 Nat Commun, DOI 10.1038/s41467-024-49616-z (PMC11228034)
- Zheng et al. 2022 Curr Biol, DOI 10.1016/j.cub.2022.06.031 (PMC9413950)
- Shuai et al. 2025 eLife, DOI 10.7554/eLife.94168. https://elifesciences.org/articles/94168/figures, supp3/supp5 xlsx
- FBbt:00048241: https://www.ebi.ac.uk/ols4/api/ontologies/fbbt/terms?obo_id=FBbt:00048241
- VFB Cha-F-000350: https://v3-cached.virtualflybrain.org/get_term_info?id=VFB_00004372
- lhns (Jefferis lab) typing tables: https://github.com/jefferislab/lhns (`data-raw/csv/hemibrain_olfactory_lateral_horn_neurons.csv`, `data-raw/lm/processLHNs.R`)
- Tanaka et al. 2008 J Comp Neurol, DOI 10.1002/cne.21692: PubMed abstract only (PMID 18395827)
