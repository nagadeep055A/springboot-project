package com.example.demo.entity;

import jakarta.persistence.*;
import java.util.ArrayList;
import java.util.List;
import com.fasterxml.jackson.annotation.JsonManagedReference;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@Entity
@Table(name = "ai_tools")
@JsonIgnoreProperties({"hibernateLazyInitializer", "handler"})
public class AiTool {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String name;
    private String shortDescription;

    @Lob
    @Column(columnDefinition = "TEXT")
    private String overview;

    @OneToMany(mappedBy = "aiTool", cascade = CascadeType.ALL, orphanRemoval = true, fetch = FetchType.EAGER)
    @JsonManagedReference
    private List<AiToolDetail> details = new ArrayList<>();

    public AiTool() {}

    // ⭐ Constructor your DataLoader needs
    public AiTool(String name, String shortDescription) {
        this.name = name;
        this.shortDescription = shortDescription;
        this.overview = "";
    }

    // Original full constructor
    public AiTool(String name, String shortDescription, String overview) {
        this.name = name;
        this.shortDescription = shortDescription;
        this.overview = overview;
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getShortDescription() { return shortDescription; }
    public void setShortDescription(String shortDescription) { this.shortDescription = shortDescription; }

    public String getOverview() { return overview; }
    public void setOverview(String overview) { this.overview = overview; }

    public List<AiToolDetail> getDetails() { return details; }
    public void setDetails(List<AiToolDetail> details) { this.details = details; }

    public void addDetail(AiToolDetail d) {
        d.setAiTool(this);
        this.details.add(d);
    }

    public void removeDetail(AiToolDetail d) {
        d.setAiTool(null);
        this.details.remove(d);
    }
}
