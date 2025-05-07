var profileId

document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('.form-container');
    const profileIdInput = document.getElementById('profile_id');
    const submitButton = document.getElementById('submit-button');

    form.addEventListener('submit', async function (event) {
        event.preventDefault();

        profileId = profileIdInput.value;

        try {
            const response = await fetch('/search_profile', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ profile_id: profileId })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            displayData(data);

        } catch (error) {
            console.error('There was a problem with the fetch operation:', error);
        }
    });

    function displayData(data) {
        console.log(data)
        const entries = Object.entries(data);

        // Hide the submit button
        submitButton.style.display = 'none';

        // Create Feedback Value section
        const feedbackSection = document.createElement('div');
        feedbackSection.className = 'form-group';
        feedbackSection.innerHTML = `
            <label for="feedback_value">Feedback Value</label>
            <input type="text" id="feedback_value" name="feedback_value">
        `;
        form.appendChild(feedbackSection);

        // Create new button element
        const newButton = document.createElement('button');
        newButton.textContent = 'Submit Feedback';
        newButton.className = 'my-button';
        newButton.onclick = async function () {
            const feedbackValue = document.getElementById('feedback_value').value;
            try {
                const response = await fetch('/update_feedback', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ feedback: feedbackValue, profile_id: profileId }),
                });

                if (response.ok) {
                    const responseData = await response.json();
                    if (responseData.success) {
                        alert('Feedback Success');
                    }
                } else {
                    alert('Upload Failed, Sorry');
                }
            } catch (error) {
                console.error('Upload Error:', error);
                alert('Something Wrong, Sorry');
            }
        };
        form.appendChild(newButton);

        // Create a refresh button
        const refreshButton = document.createElement('button');
        refreshButton.textContent = 'Reset';
        refreshButton.className = 'my-button';
        refreshButton.onclick = function () {
            location.reload(); // Forces the page to reload
        };
        form.appendChild(refreshButton);



        entries.forEach(([key, value]) => {
            // Define keys to exclude
            const excludeKeys = ['message'];

            // Continue to the next iteration if the key is in excludeKeys
            if (excludeKeys.includes(key)) {
                return;
            }
            const field = document.createElement('div');
            field.className = 'form-group';
            field.innerHTML = `<label>${key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' ')}</label>
            <input type="text" value="${value != null ? value : ''}" readonly>`;

            form.appendChild(field);
        });
    }
});