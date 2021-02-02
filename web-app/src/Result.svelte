<script>
  import Chart from 'chart.js';
  import 'chartjs-plugin-colorschemes';
  import { onMount } from 'svelte';
  import AddLabel from './AddLabel.svelte';

  export let resultData;

  let selectedDimension = 0;

  const ctx = 'myChart';
  let clientWidth;
  let windowHeight;
  $: canvasSize = Math.min(clientWidth, windowHeight * 0.8);
  let scatterChart;
  const recognizedPerson = resultData.labels[0];

  const dataTable = resultData.embeddings.map((embedding, index) => ({ label: resultData.labels[index], embedding }));
  const groupedByPerson = dataTable.reduce(
    (resultMap, datum) => resultMap.set(datum.label, [...(resultMap.get(datum.label) || []), datum.embedding]),
    new Map()
  );

  $: {
    let updatedDatasets = [];
    groupedByPerson.forEach(
      (embeddings, label) =>
        (updatedDatasets = [
          ...updatedDatasets,
          {
            label,
            data: embeddings.map((embedding) => ({
              x: embedding[selectedDimension],
              y: embedding[selectedDimension + 1],
            })),
          },
        ])
    );
    if (scatterChart) {
      scatterChart.data.datasets = updatedDatasets;
      scatterChart.options.scales.xAxes = [
        { display: true, scaleLabel: { display: true, labelString: `Dimension #${selectedDimension}` } },
      ];
      scatterChart.options.scales.yAxes = [
        { display: true, scaleLabel: { display: true, labelString: `Dimension #${selectedDimension + 1}` } },
      ];
      scatterChart.update();
    }
  }

  onMount(() => {
    if (recognizedPerson != 'Unknown') {
      scatterChart = new Chart(ctx, {
        type: 'scatter',
        data: {
          datasets: [],
        },
        options: {
          elements: {
            point: {
              pointStyle: (ctx) => (ctx.dataset.label === recognizedPerson ? 'circle' : 'star'),
              radius: (ctx) => (ctx.dataset.label === recognizedPerson ? 6 : 3),
              borderWidth: (ctx) => (ctx.dataset.label === recognizedPerson ? 6 : 3),
            },
          },

          tooltips: {
            callbacks: {
              label: (tooltipItem, data) => {
                return data.datasets[tooltipItem.datasetIndex].label;
              },
            },
          },
          legend: { display: false },
          maintainAspectRatio: false,
        },
      });
    }
  });
</script>

{#if recognizedPerson != 'Unknown'}
  <h2>Result: {recognizedPerson}</h2>
  <div class="row justify-content-center w-100" bind:clientWidth>
    <div style="width: {canvasSize}px; height: {canvasSize}px">
      <canvas id="myChart" />
    </div>
  </div>
  <label for="dimensionRange" class="form-label">Select dimensions</label>
  <input
    type="range"
    class="form-range"
    min="0"
    max={resultData.embeddings[0].length - 2}
    id="dimensionRange"
    bind:value={selectedDimension}
  />
{:else}
  <AddLabel />
{/if}

<svelte:window bind:innerHeight={windowHeight} />
