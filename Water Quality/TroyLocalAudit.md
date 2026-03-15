# Troy, NY — Local Data Audit
## ZIP 12180 Water Systems Analysis

*Source: NYPIRG "What's In My Water" tool — nypirg.org/whatsinmywater*
*March 2026 | BFOR 516 Capstone*

---

## Why This Data Matters to the Pitch

The NYPIRG tool is both **validation** and **evidence of the gap**. It proves:

1. The raw data is publicly available — the argument "government doesn't collect this" is false
2. The data exists in a form that still fails the consumer's job — a ZIP code lookup is not a tap-level safety signal
3. The fragmentation problem is visible and severe — 46+ water systems in a single ZIP code
4. The most vulnerable populations (renters, mobile home parks, senior housing, schoolchildren) are all present in this dataset and have no way to self-navigate it

**The NYPIRG tool IS the "why hasn't someone built this" answer.** NYPIRG built something. It proves the data is available. It fails the job because it answers "which system has historic violations" not "is my specific tap safe right now."

---

## ZIP 12180 — Water System Inventory

### Primary Target: Troy City PWS
| Field | Value |
|---|---|
| Population served | **51,401** |
| Source water | Surface Water |
| System type | CWS (Community Water System) |
| Lead crisis documented | 35.4 ppb at 90th percentile — higher than Flint at its worst |
| Service area | City of Troy + surrounding communities (~100,000+ total) |

**This is the beachhead system.** 51,401 people in the city alone. One utility relationship. Known lead crisis. Active federal grant. Already using AI for pipe inventory. Missing: the resident-facing safety signal.

---

### Secondary Targets: Schools (Perfect B2B2C Wedge)

| System | Pop. Served | Source | Notes |
|---|---|---|---|
| BELL TOP SCHOOL | 346 | Ground Water | NTNCWS — non-transient, non-community; students present daily |
| GRAFTON ELEM SCHOOL | 100 | Ground Water | NTNCWS |
| ROBERT C. PARKER SCHOOL | 200 | Ground Water | NTNCWS |

**Why schools are the GTM wedge:**
- Federal mandate under America's Water Infrastructure Act requires schools to test fountains and notify parents
- They are already obligated to purchase testing — your product is a better version of what they must buy
- Parent notification creates immediate community spread of your platform
- Ground water sources are unregulated by SDWA at the household level — elevated risk and zero oversight
- One school building → hundreds of parents instantly reachable via existing communication channels

---

### Vulnerable Housing — High Urgency Beachhead Residents

| System | Pop. Served | Source | Type |
|---|---|---|---|
| DIAMOND WOODS ESTATES MHP | 150 | Ground Water | CWS (mobile home park) |
| LAKESIDE GROVE MOBILE HOME PK | 120 | Ground Water | CWS |
| TERRACE HAVEN MOBILE HOME PARK | 210 | Ground Water | CWS |
| ST JUDES SEN. CITIZEN HOUSING | 56 | Ground Water | CWS |
| COTTAGE PARK APARTMENTS | 35 | Ground Water | CWS |
| PIRRI APARTMENTS | 72 | Ground Water | CWS |
| SNYDERS LAKE ROAD APARTMENTS | 54 | Ground Water | CWS |
| WILLOWBROOK APARTMENTS | 30 | Ground Water | CWS |

**Total vulnerable housing population in ZIP 12180 (est.):** ~727 people across 8 systems

**Critical pattern:** Every single one of these is on **ground water**. Ground water is more susceptible to corrosion chemistry that leaches lead from premise plumbing — and these are the lowest-income residents with the least ability to test, filter, or move. Three mobile home parks in one ZIP code. This is the beachhead in the beachhead.

---

### The Fragmentation Problem — 46 Systems in One ZIP Code

The full ZIP 12180 inventory has **46 distinct water systems**. A resident in ZIP 12180 who tries to use the NYPIRG tool faces this sequence:

1. Enter ZIP code
2. See 46 system names — most residents cannot identify their own
3. Click "view contaminants" on an assumed match
4. Receive a table of violation history in regulatory language
5. Have no idea what it means for their tap today

**This is the job that fails.** Not because the data doesn't exist — NYPIRG proved it does. But because:
- The interface assumes the user knows their water system name (most don't)
- Historic violation data ≠ real-time tap safety status
- The output is regulatory language, not plain-language guidance
- There is no address-level resolution within a system
- There is no shareable signal — each lookup lives in one browser session

**Your product resolves all five failures.** NYPIRG validated the data layer. You resolve the last mile.

---

## The NYPIRG Tool as Pitch Evidence

### How to Use It During the Pitch

> "This tool exists. NYPIRG built it. I used it in college doing environmental outreach. You enter your ZIP code and get... this."

*[Show the 46-system list on screen]*

> "46 water systems in one ZIP code. The resident doesn't know which one they're on. They see 'Brunswick Water District #2A — 35 people served.' They don't know if that's them. They click through and get a regulatory violation table from 2019.
>
> The data is there. Public. Available. Sitting in a government database NYPIRG already tapped. What doesn't exist is the address-level, plain-language, real-time signal that reaches the family at the tap.
>
> That's the gap. And that gap is why a mother in Troy still doesn't know."

This is the most credible version of "why hasn't someone built this" — because someone tried, and it still fails the job.

---

## The Brunswick Districts — A Different Story

The 14 Brunswick Water Districts in this ZIP code are a separate opportunity:
- Total served: ~13,934 people across 14 systems (Brunswick Consolidated + districts 1–16)
- All surface water sources
- Fragmented management: 14 separate systems that a resident must identify by district number

**Insight for GTM:** Brunswick residents face the same navigation problem as Troy residents — they can't easily identify which of 14 district systems they're on. A county-level deployment covering all Brunswick districts simultaneously would be a natural expansion from the Troy City beachhead.

---

## What to Pull Next (Before the Pitch)

The NYPIRG tool shows which systems to click through. Before the pitch, spend 20 minutes clicking "view contaminants" for:

1. **Troy City PWS** — document every contaminant with detected levels vs. EPA limits
2. **Bell Top School** — schools are the B2B wedge; having school-specific violation data is powerful
3. **Diamond Woods Estates MHP / Lakeside Grove / Terrace Haven** — mobile home park violations hit hardest with equity framing

Record exact contaminant names, detected levels, and EPA action levels. Those go into the prototype as real hardcoded data — not placeholders.

**URL to use:**
`https://www.nypirg.org/whatsinmywater/data/?zipcode=12180&btn=Search`

Then click "view contaminants" for each target system.

---

## Updated Prototype Data (Replace Placeholders With This)

Once you pull the real contaminant data from NYPIRG, replace the prototype's hardcoded values with actual Troy City PWS readings. Real numbers from the actual system serving the ZIP code where RPI sits — that is not a demo, that is a live evidence brief.

---

*Document created: March 2026 | Source: NYPIRG What's In My Water, ZIP 12180*
*Contaminant detail to be added after manual NYPIRG data pull*
