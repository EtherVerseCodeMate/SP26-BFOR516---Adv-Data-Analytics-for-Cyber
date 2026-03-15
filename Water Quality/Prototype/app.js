const addressData = {
    "247 River St": {
        full: "247 River St, Troy NY 12180",
        serviceLine: "UNKNOWN",
        serviceDesc: "Replacement scheduled Q3 2026. City grant active.",
        contaminants: [
            { name: "Lead", level: "2.1 ppb", limit: "15 ppb", status: "safe", verdict: "Safe for drinking", rec: "Below EPA action level. No immediate action required." },
            { name: "Iron", level: "0.38 mg/L", limit: "0.3 mg/L", status: "warn", verdict: "Elevated · Safe to drink", rec: "May cause minor staining or metallic taste. Standard filter recommended." },
            { name: "PFAS", level: "< 1 ppt", limit: "4 ppt", status: "safe", verdict: "Not detected", rec: "Levels are within safe health guidelines." }
        ]
    },
    "123 Congress St": {
        full: "123 Congress St, Troy NY 12180",
        serviceLine: "KNOWN LEAD",
        serviceDesc: "Lead service line identified. Scheduled for urgent replacement.",
        contaminants: [
            { name: "Lead", level: "9.4 ppb", limit: "15 ppb", status: "warn", verdict: "Elevated · Monitor", rec: "Approaching EPA action level. Use a certified lead-reduction filter for infants." },
            { name: "TTHMs", level: "68.2 ppb", limit: "80 ppb", status: "warn", verdict: "Passes Legal Standard", rec: "455x over health guidelines. Carbon filter highly recommended." },
            { name: "Chromium-6", level: "0.12 ppb", limit: "N/A", status: "warn", verdict: "No Legal Limit", rec: "Above health guidelines. Reverse osmosis required for full removal." }
        ]
    },
    "89 Ferry St": {
        full: "89 Ferry St, Troy NY 12180",
        serviceLine: "KNOWN LEAD",
        serviceDesc: "DANGER: Active lead service line with high concentration.",
        contaminants: [
            { name: "Lead", level: "18.2 ppb", limit: "15 ppb", status: "danger", verdict: "ABOVE ACTION LEVEL", rec: "Do not use unfiltered for drinking or cooking. Flush pipes for 2 minutes." },
            { name: "Copper", level: "1.4 mg/L", limit: "1.3 mg/L", status: "danger", verdict: "Elevated Risk", rec: "Can cause gastrointestinal issues. Contact building manager immediately." },
            { name: "PFAS", level: "5.2 ppt", limit: "4 ppt", status: "warn", verdict: "Above Health Limit", rec: "Requires specialized filtration for reduction." }
        ]
    }
};

function renderStatus(addressKey) {
    const data = addressData[addressKey];
    const container = document.getElementById('status-container');
    const label = document.getElementById('current-address-label');
    const lineStatus = document.getElementById('line-status');
    const lineDesc = document.getElementById('line-desc');

    // Update labels
    label.textContent = data.full;
    lineStatus.textContent = data.serviceLine;
    lineDesc.textContent = data.serviceDesc;

    // Clear and render cards
    container.innerHTML = '';
    data.contaminants.forEach((c, index) => {
        const card = document.createElement('div');
        card.className = `status-card ${c.status} animate-up`;
        card.style.animationDelay = `${index * 0.1}s`;
        
        card.innerHTML = `
            <div class="card-header">
                <span class="contaminant-name">${c.name}</span>
                <span class="status-badge">${c.status}</span>
            </div>
            <div class="level-info">${c.level} · Limit: ${c.limit}</div>
            <div class="verdict">${c.verdict}</div>
            <div class="recommendation">${c.rec}</div>
        `;
        container.appendChild(card);
    });

    // Update body background subtly
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

// Initial Render
window.onload = () => {
    renderStatus("247 River St");
};
