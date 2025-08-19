document.getElementById('analyze').addEventListener('click', async () => {
  const url = document.getElementById('url').value;
  const text = document.getElementById('text').value;
  const resp = await fetch('/analyze', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({url, text})
  });
  const data = await resp.json();
  const labels = Object.keys(data);
  const values = Object.values(data);
  const ctx = document.getElementById('chart').getContext('2d');
  if (window.chart) {
    window.chart.destroy();
  }
  window.chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Frequency',
        data: values,
        backgroundColor: 'rgba(54, 162, 235, 0.5)'
      }]
    }
  });
});
