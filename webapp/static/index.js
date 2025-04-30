document.addEventListener("DOMContentLoaded", function() {
    const form = document.querySelector("form");
    
    if (form) {
        form.addEventListener("submit", function(event) {
            event.preventDefault(); // 阻止默认表单提交行为
            console.log("press submit");
            const formData = new FormData(form);
            const data = {};

            formData.forEach((value, key) => {
                data[key] = value; // 将表单数据转换为 JSON
            });

            fetch('/submit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            })
            .then(response => response.json()) // 修正：解析 JSON 响应
            .then(data => {
                console.log('Success:', data);

                if (data.r !== undefined) { // 确保 r 存在
                window.location.href = `/result?r=${data.r}`;
                } else {
                    console.error('Error: Missing r in response');
                }
            })
            .catch((error) => {
                console.error('Error:', error);
                // 错误处理，可以在这里添加用户界面的错误通知
            });
        });
    }
});

document.addEventListener('DOMContentLoaded', (event) => {
    const today = new Date();
    const year = today.getFullYear();
    const month = String(today.getMonth() + 1).padStart(2, '0'); // Months are zero indexed
    const day = String(today.getDate()).padStart(2, '0');
    const dateInput = document.getElementById('date');
    
    dateInput.value = `${year}-${month}-${day}`;
});