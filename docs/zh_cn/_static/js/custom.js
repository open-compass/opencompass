var collapsedSections = ['Advanced Guides', 'Tools', 'User Guides', 'Notes'];

$(document).ready(function () {
  $('.model-summary').DataTable({
    "stateSave": false,
    "lengthChange": false,
    "pageLength": 20,
    "order": []
  });
});
