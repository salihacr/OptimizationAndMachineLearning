$(document).ready(function () {
    $(document).ajaxStart(function () {
        $("#wait").css("display", "block");
        $('#errorAlert').hide();
    });
    $(document).ajaxComplete(function () {
        $("#wait").css("display", "none");
    });
    $('#bound').keyup(function () {
        var text = $('#bound').val();
        var bounds = `(-${text}, ${text}) olarak sınırlar belirlendi`;
        $('#bounds').val(bounds);
    });

    $('#opt_form').on('submit', function (event) {

        console.log("Buttonu Disable Yaptım")
        $("#btn_run").attr("disabled", true);
        $("#btn_run").css("cursor", 'wait');

        console.log("Formdan Veri Aldım, işliyorum.")
        $("#plot_area").css("display", "none");
        var form_data = new FormData($('#opt_form')[0]);
        $.ajax({
            type: 'POST',
            url: '/optimize',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            enctype: 'multipart/form-data'
        })
            .done(function (data) {
                console.log("Adım Başarılı")
                if (data.error) {
                    console.log("HATA MEYDANA GELDİ")
                    $('#errorAlertText').text(data.error);
                    $('#errorAlert').show();
                    $("#btn_run").attr("disabled", false);
                    $("#btn_run").css("cursor", 'pointer');
                    $('#successAlert').hide();
                }
                else {
                    console.log("Hatasız Çalıştım")
                    console.log(data);

                    $('#abc_cost_graph').attr("src", data.abc_cost_path);
                    $('#sa_cost_graph').attr("src", data.sa_cost_path);
                    $('#pso_cost_graph').attr("src", data.pso_cost_path);
                    $('#de_cost_graph').attr("src", data.de_cost_path);
                    $('#compare_graph2').attr("src", data.compare_costs_path2);
                    $('#all_compare_graph').attr("src", data.all_compare_costs_path);

                    $('#all_compare_graph2').attr("src", data.all_compare_costs_path);
                    $('#all_compare_graph3').attr("src", data.all_compare_costs_path);

                    $("#plot_area").css("display", "block");

                    $('#errorAlert').hide();

                    console.log("Buttonu Enable Yaptım")
                    $("#btn_run").attr("disabled", false);
                    $("#btn_run").css("cursor", 'pointer');
                }
            });
        event.preventDefault();
    });
});
function card_checked(element) {
    console.log(element.id);
    var card = element.parentNode.parentNode;
    console.log(card);

    if (element.checked === true) {
        card.className = "card border-success";
        card.style.border = "2px solid transparent";
    } else {
        card.className = "card border";
        card.style.border = "2px solid transparent";
    }
}