<script>
  let file = undefined;

  function onFileInput({ target }) {
    file = target.files[0];
    // target.value = '';

    console.log(file);
  }

  async function handleUpload() {
    const formData = new FormData();
    formData.append('image', file, file.name);

    let response;
    response = await fetch('http://localhost:8000/recognize-face/', {
      method: 'POST',
      header: { 'Content-Type': 'multipart/form-data' },
      body: formData,
    });
    if (response.ok) {
    } else {
      throw new Error(`${response.status}: ${response.statusText}. From ${response.url}.`);
    }
  }
</script>

<div>
  <label for="formFile" class="form-label">Pick an image of a person to recognise</label>
  <input class="form-control" type="file" id="formFile" on:input={onFileInput} />

  <button type="button" class="btn btn-primary" on:click={handleUpload}>Upload</button>
</div>
