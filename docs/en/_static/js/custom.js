var collapsedSections = ['进阶教程', '工具', '教程', '其他说明'];

$(document).ready(function () {
  $('.model-summary').DataTable({
    "stateSave": false,
    "lengthChange": false,
    "pageLength": 20,
    "order": []
  });
});
