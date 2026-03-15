# Prototype Specification — WaterTruth MVP
## Change The World Weekend Challenge

*March 2026 | BFOR 516 Capstone*
*Build time estimate: 3–5 hours total*
*Judges want: concept demonstration, not production code*

---

## What the Rubric Requires

> "Does the prototype adequately demonstrate the conceptual solution?"

Minimum to max this category:
- A phone screen showing plain-language water safety status for a specific address
- The QR code flow (no install, no app, just scan)
- One sentence per contaminant: what it is, what it means for the family today
- Physical prop: QR code printed and mounted on something that looks like a tap or faucet

That is one HTML page, deployed to a free URL, with a printed QR code. **Build that first.** Everything else is bonus.

---

## Build Option A: Static HTML Page (Recommended — 2–3 hours)

### What It Is
A single `.html` file with hardcoded data for Troy, NY. No backend. No database. Deployable to GitHub Pages or Netlify in under 5 minutes.

### What It Shows
Two views:
1. **Resident View** — plain-language safety status for a specific address
2. **Building Manager View** — multi-unit summary with flags

---

### Screen 1: Resident View

```
┌─────────────────────────────────────┐
│  💧 WaterTruth                       │
│  247 River St, Troy NY 12180         │
├─────────────────────────────────────┤
│                                     │
│  📍 Your Water Status               │
│  Last updated: 2 hours ago          │
│                                     │
│  ┌─────────────────────────────┐    │
│  │  ✅  LEAD                   │    │
│  │  2.1 ppb · Below action     │    │
│  │  level · Safe for drinking  │    │
│  └─────────────────────────────┘    │
│                                     │
│  ┌─────────────────────────────┐    │
│  │  ⚠️  IRON                   │    │
│  │  0.38 mg/L · Elevated       │    │
│  │  Safe to drink · May cause  │    │
│  │  staining and taste issues  │    │
│  └─────────────────────────────┘    │
│                                     │
│  ┌─────────────────────────────┐    │
│  │  ✅  PFAS                   │    │
│  │  < 1 ppt · Not detected     │    │
│  │  Safe for drinking          │    │
│  └─────────────────────────────┘    │
│                                     │
│  ─────────────────────────────      │
│  Service line status:               │
│  ⚠️  UNKNOWN — replacement          │
│  scheduled Q3 2026                  │
│                                     │
│  Questions? Text WATER to 55512     │
│  Share this status  [🔗]            │
└─────────────────────────────────────┘
```

**Key design decisions:**
- Green checkmark = safe → judge sees the system working for the good case
- Yellow warning = elevated but not crisis → nuanced, realistic
- "Service line status: UNKNOWN" → connects directly to the trigger event in the pitch
- Share button → demonstrates the social/collective dimension of the platform
- Text line → shows no-smartphone fallback (critical for low-income beachhead)

---

### Screen 2: Building Manager View (Optional — adds 45 min)

```
┌─────────────────────────────────────┐
│  💧 WaterTruth · Manager Portal      │
│  247 River St, Troy NY 12180         │
├─────────────────────────────────────┤
│                                     │
│  Building Status: ⚠️ Monitor         │
│  Last full test: Jan 14, 2026        │
│                                     │
│  Unit   Lead    Iron    Action       │
│  ─────────────────────────────      │
│  1A     ✅ 2.1  ⚠️ 0.38  None       │
│  1B     ✅ 1.8  ✅ 0.11  None       │
│  2A     ⚠️ 9.4  ⚠️ 0.55  Retest    │
│  2B     ✅ 2.3  ✅ 0.18  None       │
│  3A     ❌ 18.2 ⚠️ 0.44  Alert      │
│                                     │
│  ⚠️ Unit 3A exceeds 15 ppb          │
│  Recommended: Notify tenant,         │
│  flush protocol, schedule retest     │
│                                     │
│  [Download Report]  [Notify Tenants] │
└─────────────────────────────────────┘
```

**Why this view matters:** It shows you understand the B2B2C model. Two users, two jobs, same data. The renter gets safety status. The manager gets liability management. This is the answer to "how does the renter get this without install rights?"

---

### HTML Template (Starter Code)

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>WaterTruth — 247 River St, Troy NY</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #f0f4f8;
      max-width: 420px;
      margin: 0 auto;
      padding: 16px;
    }
    .header {
      background: #1a56db;
      color: white;
      padding: 16px;
      border-radius: 12px;
      margin-bottom: 16px;
    }
    .header h1 { font-size: 20px; font-weight: 700; }
    .header p { font-size: 13px; opacity: 0.85; margin-top: 4px; }
    .timestamp { font-size: 12px; color: #6b7280; margin-bottom: 12px; }
    .card {
      background: white;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 12px;
      border-left: 4px solid #ccc;
    }
    .card.safe { border-left-color: #16a34a; }
    .card.warn { border-left-color: #d97706; }
    .card.danger { border-left-color: #dc2626; }
    .card-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 700;
      font-size: 16px;
      margin-bottom: 4px;
    }
    .card-level { font-size: 13px; color: #6b7280; }
    .card-verdict {
      font-size: 14px;
      font-weight: 600;
      margin-top: 6px;
    }
    .safe .card-verdict { color: #16a34a; }
    .warn .card-verdict { color: #d97706; }
    .danger .card-verdict { color: #dc2626; }
    .service-line {
      background: #fef3c7;
      border: 1px solid #fcd34d;
      border-radius: 12px;
      padding: 14px;
      margin-bottom: 12px;
      font-size: 14px;
    }
    .footer {
      text-align: center;
      font-size: 12px;
      color: #6b7280;
      margin-top: 16px;
    }
    .share-btn {
      display: block;
      background: #1a56db;
      color: white;
      border: none;
      border-radius: 8px;
      padding: 12px;
      width: 100%;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      margin-top: 16px;
    }
  </style>
</head>
<body>

  <div class="header">
    <h1>💧 WaterTruth</h1>
    <p>247 River St, Troy NY 12180</p>
  </div>

  <p class="timestamp">📍 Last updated: 2 hours ago · Troy Water Authority data</p>

  <div class="card safe">
    <div class="card-title">✅ Lead</div>
    <div class="card-level">2.1 ppb · EPA action level: 15 ppb</div>
    <div class="card-verdict">Safe for drinking and cooking</div>
  </div>

  <div class="card warn">
    <div class="card-title">⚠️ Iron</div>
    <div class="card-level">0.38 mg/L · EPA secondary standard: 0.30 mg/L</div>
    <div class="card-verdict">Safe to drink · May cause taste and staining</div>
  </div>

  <div class="card safe">
    <div class="card-title">✅ PFAS</div>
    <div class="card-level">&lt; 1 ppt · Below detection threshold</div>
    <div class="card-verdict">Not detected · Safe for drinking</div>
  </div>

  <div class="service-line">
    ⚠️ <strong>Service line status: UNKNOWN</strong><br>
    Replacement scheduled Q3 2026. Troy Water Authority will notify you when work begins on your block.<br>
    <a href="#" style="color:#92400e;">Learn what this means for your family →</a>
  </div>

  <button class="share-btn">📤 Share this status with your landlord</button>

  <div class="footer">
    Data sourced from Troy Water Authority · Updated every 4 hours<br>
    Questions? Text WATER to 55512<br><br>
    <strong>WaterTruth</strong> — Real-time water safety for your address
  </div>

</body>
</html>
```

---

## Build Option B: Address Lookup (Adds 1 hour)

Add a simple dropdown or text input that switches between 3–4 hardcoded addresses. Gives the impression of a live system without requiring a database.

```javascript
const addresses = {
  "247 River St": { lead: 2.1, iron: 0.38, pfas: "<1", lineStatus: "Unknown" },
  "123 Congress St": { lead: 9.4, iron: 0.55, pfas: "<1", lineStatus: "Known Lead" },
  "89 Ferry St": { lead: 18.2, iron: 0.44, pfas: "<1", lineStatus: "Known Lead" }
};
```

This lets you demo: enter a "good" address (safe), then enter the address with 18.2 ppb lead and watch the cards flip to red. That live state change is worth two minutes of explanation.

---

## Deployment (15 minutes)

### Option 1: GitHub Pages (Free, permanent URL)
1. Create a free GitHub account
2. New repository → upload `index.html`
3. Settings → Pages → Source: main branch
4. URL: `https://[username].github.io/watertruth`

### Option 2: Netlify Drop (Fastest — 2 minutes)
1. Go to netlify.com/drop
2. Drag and drop your `index.html` file
3. Get a live URL immediately: `https://[random-name].netlify.app`
4. Can rename to something like `watertruth-troy.netlify.app`

### Option 3: Just open the file on your phone
- Save `index.html` to your phone
- Open in mobile browser
- It will look exactly like a real app
- No deployment needed for a live demo

---

## The Physical Prop (30 minutes — HIGH IMPACT)

**This is the highest-ROI thing you can do for the prototype score.**

### What to build:
1. Generate a QR code linking to your deployed prototype URL
   - Use qr-code-generator.com or any free tool
   - Set size to at least 400x400px
2. Print it on a 4x6" card
3. Mount it on something that looks like a tap or water fixture:
   - Tape it to the side of a water bottle
   - Print it on a label and stick it to a pipe segment from a hardware store (~$2)
   - Cut a faucet shape from cardboard and attach the QR code

### During the demo:
> "A resident scans this code — it could be on their faucet, handed to them by their building super, or texted by the utility."

*[Scan it live. The prototype opens on your phone. Hand the phone to a judge.]*

> "No app download. No account. They read this."

A physical prop that a judge can hold makes the concept tangible in a way no slide can. It demonstrates the zero-friction, zero-install thesis in 10 seconds.

---

## What NOT to Build

- Authentication / login screens — adds complexity, not value for the demo
- Animated charts or complex data visualizations — judges want to see the *job*, not the data pipeline
- A 20-screen app — one perfect screen demonstrating the core job beats ten mediocre screens
- Backend API integration — static hardcoded data is fine; judges want concept, not infrastructure
- Mobile app in React Native / Flutter — too long to build, too hard to demo without install

---

## Demo Script (During Pitch — Move 5)

**Setup (5 seconds):**
*[Hold up printed QR code on physical prop]*
> "This is what it looks like at the tap."

**Scan (10 seconds):**
*[Scan QR code live or pull up prototype URL on phone]*
> "A resident scans this. No app. No account. Just this."

**Show the resident view (20 seconds):**
*[Hand phone to a judge or hold it up facing the audience]*
> "Lead: 2.1 parts per billion. Safe for drinking. Iron: elevated — safe but will stain. Service line: Unknown — replacement scheduled Q3 2026.
>
> That's the answer 22 million people don't have right now."

**If you built the address lookup (30 seconds):**
*[Type in the "bad" address]*
> "But this address, three blocks away — 18 parts per billion. Above the EPA action level. The card turns red. The recommended action appears: run your tap for 2 minutes before use, use a certified filter, contact the city."

**Show building manager view (20 seconds, if built):**
> "The building manager sees this — which units need follow-up, which residents to notify. Same data. Two different jobs."

**Return to pitch:**
> "That is the concept. The data already exists. The city already collects it. We're the translation layer."

---

## Prototype Scoring Self-Assessment

| Rubric Item | Minimum to Score | Your Build |
|---|---|---|
| Demonstrates the conceptual solution | One screen with plain-language status | ✅ Resident view |
| Not just a feature list slide | Working or mock UI that a judge can interact with | ✅ Live URL or phone |
| Core job demonstrated | Status visible without install or account | ✅ QR code flow |
| Physical artifact | Something a judge can touch | ✅ Printed QR prop |
| B2B2C model shown | Second view showing the buyer's job | ✅ Manager view (if built) |

---

## Final Build Order (Time-Boxed)

| Task | Time | Priority |
|---|---|---|
| Write and style resident view HTML | 1.5 hours | Must |
| Deploy to Netlify or GitHub Pages | 15 min | Must |
| Generate and print QR code | 15 min | Must |
| Mount QR code on physical prop | 30 min | Must |
| Add address lookup (3 hardcoded addresses) | 45 min | High |
| Build building manager view | 45 min | High |
| Add "danger" address showing red state | 30 min | High |
| Add SMS/text fallback copy | 15 min | Medium |

**Stop building when you have Must + High complete. Everything else is polish.**

---

*Prototype spec locked: March 2026 | Build before rehearsal — rehearse with the actual prototype in hand*
