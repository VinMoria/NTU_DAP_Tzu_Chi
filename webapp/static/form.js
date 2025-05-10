document.addEventListener("DOMContentLoaded", function() {
    const form = document.querySelector("form");
    
    if (form) {
        form.addEventListener("submit", function(event) {
            event.preventDefault(); // 阻止默认表单提交行为
            const formData = new FormData(form);
            const data = {};

            formData.forEach((value, key) => {
                if (key === "custom_input" && value === "0") { // 检查是否为默认值
                    data[key] = -1;
                } else {
                    data[key] = value; // 将表单数据转换为 JSON
                }
                console.log(key, value)
            });
            data["points"] = countCheckedCheckboxes()
            data["special_cases"] = getCheckedLabels()

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
                window.location.href = `/result?r=${data.r}&profile_id=${data.profile_id}&hint=${data.hint}`;
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

function toggleInput() {
    const defaultCheck = document.getElementById('default_check');
    const customInput = document.getElementById('custom_input');
    if (defaultCheck.checked) {
        customInput.disabled = true;
        customInput.value = 0; // 设定为默认数字值
    } else {
        customInput.disabled = false;
        customInput.value = ''; // 清除输入框
    }
}

function countCheckedCheckboxes() {
    const checkboxes = document.querySelectorAll('input[name="special_case"]:checked');
    return checkboxes.length;
}

function getCheckedLabels() {
    const checkboxes = document.querySelectorAll('input[name="special_case"]:checked');
    const labels = Array.from(checkboxes).map(checkbox => {
        return document.querySelector(`label[for="${checkbox.id}"]`).textContent;
    });
    return labels.join(', ');
}