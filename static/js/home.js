$(function () {
    $('#submits').click(function () {
        var content = $("#resume_input").val();
        console.log('js显示：',content);

        $.ajax({
            url: "/single_extract_form",
            type: 'POST',
            contentType: "application/json; charset=utf-8",
            dataType: 'json',
            data: JSON.stringify({"resume":content}),
//            data: {resume:content},

            success: function (result) {
                if(result.code == 1){
                  console.log('后端返回数据',result.data);
                }
                else {
                    console.log('后端处理失败')
                }
            },
            error: function () {
                alert('提交数据出错')
            }
        });
    });
});

