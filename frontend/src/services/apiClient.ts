import { toast } from 'react-toastify';

async function apiFetch(endpoint, options) {
    try {
        const response = await fetch(`${API_URL}${endpoint}`, options);
        if (!response.ok) throw new Error('API error');
        return await response.json();
    } catch (error) {
        toast.error(`Error: ${error.message}`);
        throw error;
    }
}
