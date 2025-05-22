import React from "react";
import PropTypes from "prop-types";
import { Card, Typography, List, ListItem, ListItemText } from "@mui/material";

export default function GrokFeedback({ feedback, loading }) {
  if (loading) return <Card sx={{ p: 2, mb: 2 }}>Analyzing with Grok...</Card>;
  if (!feedback || feedback.length === 0) return null;
  return (
    <Card sx={{ p: 2, mb: 2 }}>
      <Typography variant="h6" gutterBottom>Grok AI Suggestions</Typography>
      <List>
        {feedback.map((item, idx) => (
          <ListItem key={idx}>
            <ListItemText primary={item} />
          </ListItem>
        ))}
      </List>
    </Card>
  );
}

GrokFeedback.propTypes = {
  feedback: PropTypes.arrayOf(PropTypes.string),
  loading: PropTypes.bool,
};
