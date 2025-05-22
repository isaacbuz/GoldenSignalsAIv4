describe('API Security & Contract', () => {
  it('should return OpenAPI schema', () => {
    cy.request('/openapi.json').then((response) => {
      expect(response.status).to.eq(200);
      expect(response.body).to.have.property('paths');
      expect(response.body).to.have.property('components');
    });
  });

  it('should enforce CORS headers', () => {
    cy.request({
      method: 'OPTIONS',
      url: '/api/some-endpoint',
      headers: {
        Origin: 'https://yourfrontend.com',
        'Access-Control-Request-Method': 'GET',
      },
    }).then((response) => {
      expect(response.headers['access-control-allow-origin']).to.eq('https://yourfrontend.com');
    });
  });

  it('should enforce security headers', () => {
    cy.request('/api/some-endpoint').then((response) => {
      expect(response.headers).to.have.property('x-frame-options');
      expect(response.headers).to.have.property('x-content-type-options');
    });
  });
});
