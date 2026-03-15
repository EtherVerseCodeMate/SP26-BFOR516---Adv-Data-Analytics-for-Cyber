const addressData = {
    "247 River St": {
        full: "247 River St, Troy NY 12180",
        serviceLine: "UNKNOWN",
        serviceLineStatus: "warn",
        serviceDesc: "Replacement scheduled Q3 2026. City grant active.",
        filterAdvisory: null,
        contaminants: [
            { name: "Lead",  level: "2.1 ppb",   limit: "15 ppb",  status: "safe",   verdict: "Safe for drinking",    rec: "Below EPA action level. No immediate action required." },
            { name: "Iron",  level: "0.38 mg/L",  limit: "0.3 mg/L", status: "warn",  verdict: "Elevated · Safe to drink", rec: "May cause minor staining or metallic taste. Standard filter recommended." },
            { name: "PFAS",  level: "< 1 ppt",   limit: "4 ppt",   status: "safe",   verdict: "Not detected",         rec: "Levels are within safe health guidelines." }
        ]
    },
    "123 Congress St": {
        full: "123 Congress St, Troy NY 12180",
        serviceLine: "KNOWN LEAD",
        serviceLineStatus: "danger",
        serviceDesc: "Lead service line identified. Scheduled for urgent replacement.",
        filterAdvisory: "Standard pitcher filters (Brita, PUR) reduce TTHMs and HAAs but do <strong>not</strong> remove Chromium-6 or radium. Reverse osmosis removes all contaminants above health guidelines.",
        contaminants: [
            { name: "Lead",         level: "9.4 ppb",   limit: "15 ppb",     status: "warn",   verdict: "Elevated · Monitor",        rec: "Approaching EPA action level. Use a certified lead-reduction filter — especially for infants and children." },
            { name: "TTHMs",        level: "68.2 ppb",  limit: "80 ppb",     status: "warn",   verdict: "Passes Legal Standard",     rec: "455× over health guidelines. Carbon filter highly recommended." },
            { name: "Chromium-6",   level: "0.12 ppb",  limit: "No limit",   status: "warn",   verdict: "No Federal Limit · Elevated", rec: "Above health risk guideline. Requires reverse osmosis for full removal." }
        ]
    },
    "89 Ferry St": {
        full: "89 Ferry St, Troy NY 12180",
        serviceLine: "KNOWN LEAD",
        serviceLineStatus: "danger",
        serviceDesc: "ACTIVE ALERT: Lead service line confirmed. High concentration detected.",
        filterAdvisory: "Standard pitcher filters do <strong>not</strong> reliably remove lead at these concentrations. Use an NSF/ANSI 53-certified filter or reverse osmosis. Do not use unfiltered tap water for drinking, cooking, or infant formula.",
        contaminants: [
            { name: "Lead",   level: "18.2 ppb", limit: "15 ppb",   status: "danger", verdict: "ABOVE ACTION LEVEL",   rec: "Do not use unfiltered for drinking or cooking. Flush cold water for 2 minutes before use. Contact building manager immediately." },
            { name: "Copper", level: "1.4 mg/L", limit: "1.3 mg/L", status: "danger", verdict: "Elevated Risk",        rec: "Can cause gastrointestinal issues. Contact your building manager immediately." },
            { name: "PFAS",   level: "5.2 ppt",  limit: "4 ppt",    status: "warn",   verdict: "Above Health Limit",   rec: "Exceeds EPA health advisory. Requires specialized filtration for reduction." }
        ]
    }
};

const statusEmoji = { safe: '✅', warn: '⚠️', danger: '❌' };
const statusLabel = { safe: 'SAFE', warn: 'ELEVATED', danger: 'DANGER' };

function renderStatus(addressKey) {
    const data = addressData[addressKey];
    const container = document.getElementById('status-container');
    const label = document.getElementById('current-address-label');
    const lineStatus = document.getElementById('line-status');
    const lineDesc = document.getElementById('line-desc');
    const serviceInfo = document.getElementById('service-info');

    label.textContent = data.full;
    lineStatus.textContent = data.serviceLine;
    lineDesc.textContent = data.serviceDesc;

    // Color the service line section based on severity
    serviceInfo.className = 'service-info ' + data.serviceLineStatus;

    container.innerHTML = '';

    data.contaminants.forEach((c, index) => {
        const card = document.createElement('div');
        card.className = `status-card ${c.status} animate-up`;
        card.style.animationDelay = `${index * 0.1}s`;
        card.innerHTML = `
            <div class="card-header">
                <span class="contaminant-name">${statusEmoji[c.status]} ${c.name}</span>
                <span class="status-badge">${statusLabel[c.status]}</span>
            </div>
            <div class="level-info">${c.level} · Limit: ${c.limit}</div>
            <div class="verdict">${c.verdict}</div>
            <div class="recommendation">${c.rec}</div>
        `;
        container.appendChild(card);
    });

    // Filter advisory box (only shown when relevant)
    if (data.filterAdvisory) {
        const advisory = document.createElement('div');
        advisory.className = 'filter-advisory';
        advisory.innerHTML = `<strong>⚠️ Note on your filter:</strong><br>${data.filterAdvisory}`;
        container.appendChild(advisory);
    }

    // Tint page background to match worst status
    if (data.contaminants.some(c => c.status === 'danger')) {
        document.body.style.backgroundColor = '#fff1f2';
    } else if (data.contaminants.some(c => c.status === 'warn')) {
        document.body.style.backgroundColor = '#fffbeb';
    } else {
        document.body.style.backgroundColor = '#f8fafc';
    }
}

function handleAddressChange() {
    const selector = document.getElementById('address-search');
    renderStatus(selector.value);
}

function switchView(view) {
    const resView = document.getElementById('resident-view');
    const manView = document.getElementById('manager-view');
    const resBtn = document.getElementById('btn-resident');
    const manBtn = document.getElementById('btn-manager');

    if (view === 'resident') {
        resView.style.display = 'block';
        manView.style.display = 'none';
        resBtn.classList.add('active');
        manBtn.classList.remove('active');
    } else {
        resView.style.display = 'none';
        manView.style.display = 'block';
        resBtn.classList.remove('active');
        manBtn.classList.add('active');
    }
}

window.onload = () => {
    renderStatus("247 River St");
};
