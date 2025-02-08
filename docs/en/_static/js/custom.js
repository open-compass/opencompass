var collapsedSections = ['Dataset Statistics'];

$(document).ready(function () {
  $('.dataset').DataTable({
    "stateSave": false,
    "lengthChange": false,
    "pageLength": 20,
    "order": [],
    "language": {
      "info": "Show _START_ to _END_ Items（Totally _TOTAL_ ）",
      "infoFiltered": "（Filtered from _MAX_ Items）",
      "search": "Search：",
      "zeroRecords": "Item Not Found",
      "paginate": {
        "next": "Next",
        "previous": "Previous"
      },
    }
  });
});
