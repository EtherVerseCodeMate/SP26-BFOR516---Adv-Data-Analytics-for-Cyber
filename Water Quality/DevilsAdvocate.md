# Devil's Advocate: What You're Not Seeing Yet

*March 2026 | BFOR 516 Capstone — Water Quality Pitch Prep*

---

## Blind Spot #1: This Problem Is Already Being Solved

Before your pitch, you need to know these exist and be ready to answer "how are you different?":

| Company | What They Do | Why It Matters to You |
|---|---|---|
| **Flo by Moen** | Smart water monitor, leak + quality detection, $500 installed | Already in homes, VC-backed, mainstream |
| **Hach / YSI** | Real-time municipal water quality sensors | Industrial scale, utilities already buying |
| **Tap Score (SimpleLab)** | Mail-in testing + plain-language results, ~$69–$300 | Directly addresses the "confusing lab report" problem |
| **Bluefield Research** | AI water analytics for utilities | B2B, but same data layer |
| **WaterSmart** | Utility customer engagement platform | Already inside the utility relationship |
| **Phyn** | Ultrasonic in-pipe sensor, lead detection in development | Funded, at-home, pipe-native |

**The honest question:** Are you building something genuinely new, or a better UI on top of what already exists? That's not disqualifying — but judges WILL ask.

### How to Answer "Tap Score Already Does This"

Tap Score is a snapshot (mail-in, wait days, one result). Your angle is **continuous + real-time + zero installation**. A snapshot tells you what your water was on Tuesday. You're answering what it is *right now*, at 7am, when you're making formula. Completely different job.

### How to Answer "Flo by Moen Already Does This"

Flo costs $500 installed, requires professional plumbing, and is marketed to homeowners protecting property from leaks. Your beachhead is a renter with no install rights and $60/month for bottled water. Different buyer, different budget, different job. Flo is the golf cart. You're the bus.

---

## Blind Spot #2: The Renter Problem Is Actually a Distribution Problem

Your beachhead customer — the low-income renter in a pre-1986 building — has a fatal adoption barrier nobody in your docs has named yet:

> **She cannot install anything. She may not have a spare $50. Her landlord will remove unauthorized devices. And she may not trust another institution telling her something about her water.**

The person with the highest urgency has the lowest ability to adopt a hardware solution. This is a classic "last mile" trap.

**Possible exits from this trap:**

| Approach | How It Resolves the Barrier |
|---|---|
| No-hardware model (data synthesis only) | No install needed — pull from utility SCADA, city monitoring stations, building permit records, census pipe age data |
| Landlord / property manager as buyer | Shift the install responsibility to the entity with legal access to the building |
| City/utility as distribution partner | Utility pushes alerts to residents — you sell the data layer to the utility, not the renter |
| Community organization partnerships | Schools, clinics, tenant orgs absorb the cost and distribute access |

The no-hardware path is the most defensible for the competition because it removes the adoption barrier entirely while keeping the consumer-facing job statement intact.

---

## Blind Spot #3: "Change The World" Requires a Systems Lens

This competition is not "build a better filter app." The judges (Kiran Uppuluri, vertexD) are thinking in systems:

- *"The ones who win will be the ones who understand"*
- *"What does their life look like when it is done?"*

The truly world-changing angle isn't "tell people their water is bad." It's:

> **What happens when 21 million people have irrefutable, real-time data about their water quality — and can share it collectively?**

That's a civic infrastructure play. That's environmental justice at scale. That's a dataset that:
- Forces utilities to stop hiding behind lagging annual averages
- Gives regulators the real-time view they don't currently have
- Gives lawyers and advocacy organizations evidence for enforcement
- Gives mothers in Newark the same data the EPA has — or better

**That** changes the world. A consumer app doesn't.

---

## Blind Spot #4: Your Solution Framing May Be Too Narrow

The whiteboard brainstorm showed something bigger than a consumer app:

```
Water Quality Management / Administration
    → Existing infrastructure
        → Delivery
            → Clean water
```

That's a **systems redesign**. The ML pipeline + existing pipe infrastructure + connectivity angle suggests embedding intelligence *into* infrastructure — not selling another device.

**The more defensible and scalable angle: sell to cities/utilities, not renters.**

- Utilities are already under EPA enforcement pressure
- They have budget (infrastructure bills, compliance fines)
- They have the install rights your renter doesn't
- They have the distribution channel to every ratepayer
- Your consumer-facing value proposition becomes a utility *feature* — "SafeWater alerts from [City Water Authority]"

This is a classic B2B2C play. You win by selling to the entity with money and access, and deliver value to the person who needs it most.

---

## Blind Spot #5: The 10-Year EPA Timeline Is Your Enemy — Unless You Reframe It

Current framing: the 10-year replacement window = opportunity (families exposed during the gap).

The judge's counter: *"If government is already mandating full replacement by 2034, why would a utility buy a monitoring solution? They're already committed to the fix."*

**Your reframe:**
> "Because families are exposed for the next 10 years, and utilities have no way to communicate real-time safety status to residents *during* the replacement window. The replacement is happening street by street, building by building. A family two blocks from the completed replacement doesn't know if their line is done yet. We solve the communication gap during the transition — the exact window where liability, trust, and public health risk are highest."

The wedge is the **transition period**, not the end state. Utilities face maximum reputational and legal risk during replacement. That's when they most need this.

---

## Blind Spot #6: The Two-Betrayal Narrative Has a Third Act

From the synthesis: Government betrayal + market betrayal (Brita doesn't remove lead).

The third act judges haven't heard: **information betrayal.**

The annual Consumer Confidence Report (CCR) is *legally required* to be mailed to every ratepayer. It is written by lawyers for legal compliance. It uses system-wide averages that legally comply while individual buildings may be spiking. The document that is supposed to protect the public is specifically designed to not alarm them.

That's not just a gap — it's a structural information failure built into the regulatory system. Your platform doesn't just fill a gap. It **corrects a designed failure**.

---

## The Sharper "Change The World" Frame

Instead of: *"Help families know if their water is safe"*

Try: **"Build the real-time water safety data layer that individuals, utilities, and regulators are all missing — so that for the first time, a mother in Newark and the EPA in Washington are looking at the same truth."**

That's a platform. That changes the world.

---

## Q&A Gauntlet — Answer These Before You Walk In

| Judge Question | Your Answer |
|---|---|
| "Tap Score already does this — how are you different?" | Tap Score is a snapshot (mail-in, days to result). We're answering what's in your water *right now*, continuously, with no sample required. Different job entirely. |
| "Your target customer is a renter with no install rights — how do they use this?" | We don't sell to the renter. We sell to the utility or city. The renter gets access through their water authority — the same entity that already communicates with them via water bills. |
| "If cities replace all lead pipes by 2034, what's your business in 2035?" | Lead pipes are one contaminant. PFAS has no replacement deadline — EPA just set the MCL in 2024 and there's no infrastructure fix on the horizon. Climate events are increasing contamination events (wildfires, floods, agriculture runoff). Real-time water quality monitoring is a permanent infrastructure need, not a lead-era product. |
| "Why hasn't the utility or the government built this already?" | Utilities are incentivized to report compliance, not to surface problems. The CCR is designed by lawyers, not public health communicators. Government monitoring happens at the treatment plant, not at the tap. The data exists in fragments — SCADA systems, building permits, pipe age records, utility test results — but nobody has assembled it into a household-level, real-time, plain-language signal. That's the gap. |

---

## Competitive Moat Candidates (For Later, But Think Now)

| Moat Type | What It Looks Like Here |
|---|---|
| Data network effect | More homes/sensors → better predictive model → better accuracy → more adoption → more data |
| Regulatory relationships | If EPA or state agencies use your data layer, competitors can't easily displace you |
| Utility contracts | Long-term SaaS contracts with water utilities create switching costs |
| First mover in EJ compliance | Biden/Biden-era EPA prioritized environmental justice — EJ data layer has policy tailwind |
| Patent (your whiteboard note) | If the sensor/ML architecture is novel, protect it — but don't lead with this in a JTBD pitch |

---

## Updated Document Index

| File | Contents |
|---|---|
| [WaterQuality.md](WaterQuality.md) | Original framework (95% Problem / 5% Solution) |
| [MarketResearch.md](MarketResearch.md) | Full secondary research — data, health outcomes, EJ stats, workaround analysis |
| [BeachheadSynthesis.md](BeachheadSynthesis.md) | Resolves Rural vs. Municipal tension, pitch-ready framing |
| [DevilsAdvocate.md](DevilsAdvocate.md) | This document — competitive landscape, blind spots, Q&A gauntlet |

---

*Document created: March 2026 | BFOR 516 Capstone Research*
