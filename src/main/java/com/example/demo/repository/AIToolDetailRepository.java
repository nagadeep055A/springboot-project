package com.example.demo.repository;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

import com.example.demo.entity.AiToolDetail;

public interface AIToolDetailRepository extends JpaRepository<AiToolDetail, Long> {
    List<AiToolDetail> findByAiToolId(Long toolId);
}

