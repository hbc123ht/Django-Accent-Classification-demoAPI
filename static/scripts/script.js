function getMessages(){
    if (!scrolling) {
        $.get('/messages/', function(messages){
            $('#msg-list').html(messages);
        });
    }
    scrolling = false;
}
var scrolling = false;
$(function(){
    $('#msg-list-div').on('scroll', function(){
        scrolling = true;
    });
    refreshTimer = setInterval(getMessages, 500);
});