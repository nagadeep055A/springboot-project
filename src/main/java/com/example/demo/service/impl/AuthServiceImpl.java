package com.example.demo.service.impl;

import com.example.demo.entity.User;
import com.example.demo.repository.UserRepository;
import com.example.demo.service.AuthService;
import org.springframework.stereotype.Service;

@Service
public class AuthServiceImpl implements AuthService {

    private final UserRepository repo;

    public AuthServiceImpl(UserRepository repo) {
        this.repo = repo;
    }

    @Override
    public String register(User user) {

        User existing = repo.findByEmail(user.getEmail());

        if (existing != null) {
            return "Email already exists";
        }

        repo.save(user);   // Saves successfully
        return "Registered Successfully";
    }

    @Override
    public String login(String email, String password) {

        User existing = repo.findByEmail(email);

        if (existing == null) {
            return "User not found";
        }

        if (!existing.getPassword().equals(password)) {
            return "Incorrect password";
        }

        return "Login Successful";
    }
}
