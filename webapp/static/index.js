document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('Form').addEventListener('submit', function(event) {
        event.preventDefault();
        

    });
});

document.addEventListener('DOMContentLoaded', (event) => {
    const today = new Date();
    const year = today.getFullYear();
    const month = String(today.getMonth() + 1).padStart(2, '0'); // Months are zero indexed
    const day = String(today.getDate()).padStart(2, '0');
    const dateInput = document.getElementById('date');
    
    dateInput.value = `${year}-${month}-${day}`;
});