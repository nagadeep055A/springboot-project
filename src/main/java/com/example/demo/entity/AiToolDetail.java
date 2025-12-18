package com.example.demo.entity;

import jakarta.persistence.*;
import com.fasterxml.jackson.annotation.JsonBackReference;

@Entity
@Table(name = "ai_tool_details")
public class AiToolDetail {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String sectionTitle;

    @Lob
    @Column(columnDefinition = "TEXT")
    private String content;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "aitool_id")
    @JsonBackReference
    private AiTool aiTool;

    public AiToolDetail() {}

    public AiToolDetail(String sectionTitle, String content) {
        this.sectionTitle = sectionTitle;
        this.content = content;
    }

    // ⭐ REQUIRED BY DATALOADER
    public AiToolDetail(String sectionTitle, String content, AiTool aiTool) {
        this.sectionTitle = sectionTitle;
        this.content = content;
        this.aiTool = aiTool;
    }

    // getters / setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getSectionTitle() { return sectionTitle; }
    public void setSectionTitle(String sectionTitle) { this.sectionTitle = sectionTitle; }

    public String getContent() { return content; }
    public void setContent(String content) { this.content = content; }

    public AiTool getAiTool() { return aiTool; }
    public void setAiTool(AiTool aiTool) { this.aiTool = aiTool; }
}
