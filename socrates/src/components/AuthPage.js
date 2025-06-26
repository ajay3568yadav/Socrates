import React, { useState, useEffect } from 'react';
import { createClient } from '@supabase/supabase-js';

import supabase from '../config/supabaseClient'; // Adjust the import path as necessary

const AuthPage = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const [errors, setErrors] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);

  // CSS styles
  const styles = {
    container: {
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    },
    card: {
      background: 'white',
      borderRadius: '20px',
      boxShadow: '0 25px 50px rgba(0, 0, 0, 0.15)',
      overflow: 'hidden',
      width: '100%',
      maxWidth: '900px',
      display: 'flex',
      flexDirection: 'row'
    },
    formSection: {
      width: '50%',
      padding: '50px 40px'
    },
    illustrationSection: {
      width: '50%',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '50px 40px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: 'white',
      textAlign: 'center'
    },
    title: {
      fontSize: '32px',
      fontWeight: '700',
      color: '#2d3748',
      marginBottom: '8px'
    },
    subtitle: {
      fontSize: '16px',
      color: '#718096',
      marginBottom: '40px'
    },
    formGroup: {
      marginBottom: '20px'
    },
    label: {
      display: 'block',
      fontSize: '14px',
      fontWeight: '500',
      color: '#374151',
      marginBottom: '8px'
    },
    input: {
      width: '100%',
      padding: '12px 16px',
      border: '2px solid #e2e8f0',
      borderRadius: '10px',
      fontSize: '16px',
      transition: 'all 0.2s ease',
      outline: 'none',
      boxSizing: 'border-box'
    },
    inputFocus: {
      borderColor: '#667eea',
      boxShadow: '0 0 0 3px rgba(102, 126, 234, 0.1)'
    },
    inputError: {
      borderColor: '#e53e3e'
    },
    errorText: {
      color: '#e53e3e',
      fontSize: '12px',
      marginTop: '4px'
    },
    button: {
      width: '100%',
      padding: '14px 20px',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      color: 'white',
      border: 'none',
      borderRadius: '10px',
      fontSize: '16px',
      fontWeight: '600',
      cursor: 'pointer',
      transition: 'all 0.2s ease',
      marginBottom: '20px'
    },
    buttonHover: {
      transform: 'translateY(-2px)',
      boxShadow: '0 10px 20px rgba(102, 126, 234, 0.3)'
    },
    buttonDisabled: {
      opacity: '0.6',
      cursor: 'not-allowed',
      transform: 'none'
    },
    switchText: {
      textAlign: 'center',
      color: '#718096'
    },
    switchButton: {
      background: 'none',
      border: 'none',
      color: '#667eea',
      fontWeight: '600',
      cursor: 'pointer',
      textDecoration: 'underline',
      marginLeft: '4px'
    },
    illustrationIcon: {
      width: '120px',
      height: '120px',
      background: 'rgba(255, 255, 255, 0.2)',
      borderRadius: '50%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '60px',
      margin: '0 auto 30px'
    },
    illustrationTitle: {
      fontSize: '28px',
      fontWeight: '700',
      marginBottom: '16px'
    },
    illustrationText: {
      fontSize: '16px',
      opacity: '0.9',
      lineHeight: '1.6',
      marginBottom: '30px'
    },
    featureList: {
      display: 'flex',
      flexDirection: 'column',
      gap: '10px'
    },
    feature: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '8px',
      fontSize: '14px',
      opacity: '0.8'
    },
    spinner: {
      width: '20px',
      height: '20px',
      border: '2px solid transparent',
      borderTop: '2px solid white',
      borderRadius: '50%',
      animation: 'spin 1s linear infinite',
      marginRight: '8px'
    },
    loadingContent: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center'
    },
    successCard: {
      background: 'white',
      borderRadius: '20px',
      boxShadow: '0 25px 50px rgba(0, 0, 0, 0.15)',
      padding: '50px 40px',
      width: '100%',
      maxWidth: '400px',
      textAlign: 'center'
    },
    successIcon: {
      width: '80px',
      height: '80px',
      background: 'linear-gradient(135deg, #48bb78 0%, #38a169 100%)',
      borderRadius: '50%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: 'white',
      fontSize: '30px',
      fontWeight: 'bold',
      margin: '0 auto 20px'
    },
    successTitle: {
      fontSize: '24px',
      fontWeight: '700',
      color: '#2d3748',
      marginBottom: '8px'
    },
    successText: {
      color: '#718096',
      marginBottom: '30px'
    },
    userInfo: {
      background: '#f7fafc',
      borderRadius: '10px',
      padding: '20px',
      marginBottom: '30px'
    },
    logoutButton: {
      width: '100%',
      padding: '12px 20px',
      background: '#e53e3e',
      color: 'white',
      border: 'none',
      borderRadius: '10px',
      fontSize: '16px',
      fontWeight: '600',
      cursor: 'pointer',
      transition: 'background-color 0.2s ease'
    }
  };

  // Check for existing session on component mount
  useEffect(() => {
    checkUserSession();
  }, []);

  const checkUserSession = async () => {
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (session) {
        setUser({
          id: session.user.id,
          email: session.user.email,
          username: session.user.user_metadata?.username || 'User'
        });
        setIsAuthenticated(true);
      }
    } catch (error) {
      console.error('Error checking session:', error);
    }
  };
  // Add keyframes for spinner animation and auth state listener
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
      @media (max-width: 768px) {
        .auth-card {
          flex-direction: column !important;
        }
        .form-section, .illustration-section {
          width: 100% !important;
        }
      }
    `;
    document.head.appendChild(style);

    // Listen for auth state changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (event, session) => {
        if (event === 'SIGNED_IN' && session) {
          setUser({
            id: session.user.id,
            email: session.user.email,
            username: session.user.user_metadata?.username || 'User'
          });
          setIsAuthenticated(true);
        } else if (event === 'SIGNED_OUT') {
          setUser(null);
          setIsAuthenticated(false);
        }
      }
    );

    return () => {
      document.head.removeChild(style);
      subscription.unsubscribe();
    };
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const validateEmail = (email) => {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  };

  const validatePassword = (password) => {
    // More comprehensive password validation for Supabase
    const hasMinLength = password.length >= 8;
    const hasUppercase = /[A-Z]/.test(password);
    const hasLowercase = /[a-z]/.test(password);
    const hasNumbers = /\d/.test(password);
    
    return hasMinLength && hasUppercase && hasLowercase && hasNumbers;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const newErrors = {};

    if (isLogin) {
      // Login validation
      if (!formData.email) {
        newErrors.email = 'Email is required';
      } else if (!validateEmail(formData.email)) {
        newErrors.email = 'Please enter a valid email';
      }

      if (!formData.password) {
        newErrors.password = 'Password is required';
      }
    } else {
      // Signup validation
      if (!formData.username) {
        newErrors.username = 'Username is required';
      } else if (formData.username.length < 3) {
        newErrors.username = 'Username must be at least 3 characters';
      }

      if (!formData.email) {
        newErrors.email = 'Email is required';
      } else if (!validateEmail(formData.email)) {
        newErrors.email = 'Please enter a valid email';
      }

      if (!formData.password) {
        newErrors.password = 'Password is required';
      } else if (!validatePassword(formData.password)) {
        newErrors.password = 'Password must be at least 8 characters with uppercase, lowercase, and numbers';
      }

      if (!formData.confirmPassword) {
        newErrors.confirmPassword = 'Please confirm your password';
      } else if (formData.password !== formData.confirmPassword) {
        newErrors.confirmPassword = 'Passwords do not match';
      }
    }

    setErrors(newErrors);

    if (Object.keys(newErrors).length === 0) {
      setIsLoading(true);
      
      try {
        if (isLogin) {
          // Handle Login
          console.log('Attempting login with:', { email: formData.email });
          
          const { data, error } = await supabase.auth.signInWithPassword({
            email: formData.email.trim(),
            password: formData.password,
          });

          if (error) {
            console.error('Login error:', error);
            setErrors({ submit: error.message });
          } else if (data.user) {
            // Fetch user profile from User table
            const { data: userProfile, error: profileError } = await supabase
              .from('User')
              .select('username')
              .eq('id', data.user.id)
              .single();

            if (profileError) {
              console.error('Profile fetch error:', profileError);
            }

            setUser({
              id: data.user.id,
              email: data.user.email,
              username: userProfile?.username || 'User'
            });
            setIsAuthenticated(true);
          }
        } else {
          // Handle Signup with better error logging
          const signupData = {
            email: formData.email.trim(),
            password: formData.password,
            options: {
              data: {
                username: formData.username.trim()
              }
            }
          };
          
          console.log('Attempting signup with:', { 
            email: signupData.email, 
            username: signupData.options.data.username,
            passwordLength: signupData.password.length 
          });

          const { data, error } = await supabase.auth.signUp(signupData);

          if (error) {
            console.error('Signup error:', error);
            console.error('Error details:', {
              message: error.message,
              status: error.status,
              details: error
            });
            
            // Handle specific error cases
            if (error.message.includes('email')) {
              setErrors({ submit: 'Invalid email format or email already exists' });
            } else if (error.message.includes('password')) {
              setErrors({ submit: 'Password must be at least 8 characters with uppercase, lowercase, and numbers' });
            } else {
              setErrors({ submit: `Signup failed: ${error.message}` });
            }
          } else if (data.user) {
            console.log('Signup successful, user created:', data.user.id);
            
            // Insert user data into User table
            const insertData = {
              id: data.user.id,
              username: formData.username.trim(),
              email: formData.email.trim(),
              created_at: new Date().toISOString()
            };
            
            console.log('Inserting user profile:', insertData);

            const { error: insertError } = await supabase
              .from('User')
              .insert([insertData]);

            if (insertError) {
              console.error('Error inserting user profile:', insertError);
              setErrors({ submit: 'Account created but profile setup failed. Please contact support.' });
            } else {
              console.log('User profile inserted successfully');
              
              setUser({
                id: data.user.id,
                email: formData.email,
                username: formData.username
              });
              setIsAuthenticated(true);
              
              // Show success message for email verification if needed
              if (!data.session) {
                setErrors({ submit: 'Account created successfully! Please check your email to verify your account.' });
              }
            }
          }
        }
      } catch (error) {
        console.error('Unexpected authentication error:', error);
        setErrors({ submit: 'An unexpected error occurred. Please try again.' });
      }
      
      setIsLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await supabase.auth.signOut();
      setIsAuthenticated(false);
      setUser(null);
      setFormData({
        username: '',
        email: '',
        password: '',
        confirmPassword: ''
      });
    } catch (error) {
      console.error('Error logging out:', error);
    }
  };

  const switchMode = () => {
    setIsLogin(!isLogin);
    setErrors({});
    setFormData({
      username: '',
      email: '',
      password: '',
      confirmPassword: ''
    });
  };

  if (isAuthenticated) {
    return (
      <div style={styles.container}>
        <div style={styles.successCard}>
          <div style={styles.successIcon}>✓</div>
          <h2 style={styles.successTitle}>Welcome!</h2>
          <p style={styles.successText}>You're successfully logged in</p>
          
          <div style={styles.userInfo}>
            <p style={{ fontSize: '14px', color: '#718096', marginBottom: '4px' }}>
              Logged in as:
            </p>
            <p style={{ fontWeight: '600', color: '#2d3748' }}>{user.email}</p>
            {user.username && (
              <p style={{ fontSize: '14px', color: '#718096' }}>@{user.username}</p>
            )}
          </div>

          <button
            onClick={handleLogout}
            style={styles.logoutButton}
            onMouseOver={(e) => e.target.style.backgroundColor = '#c53030'}
            onMouseOut={(e) => e.target.style.backgroundColor = '#e53e3e'}
          >
            Logout
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.card} className="auth-card">
        {/* Form Section */}
        <div style={styles.formSection} className="form-section">
          <div>
            <h2 style={styles.title}>
              {isLogin ? 'Welcome Back' : 'Create Account'}
            </h2>
            <p style={styles.subtitle}>
              {isLogin 
                ? 'Sign in to your account to continue' 
                : 'Join us and start your journey today'
              }
            </p>
          </div>

          <div>
            {!isLogin && (
              <div style={styles.formGroup}>
                <label style={styles.label}>Username</label>
                <input
                  type="text"
                  name="username"
                  value={formData.username}
                  onChange={handleInputChange}
                  style={{
                    ...styles.input,
                    ...(errors.username ? styles.inputError : {})
                  }}
                  placeholder="Enter your username"
                />
                {errors.username && (
                  <p style={styles.errorText}>{errors.username}</p>
                )}
              </div>
            )}

            <div style={styles.formGroup}>
              <label style={styles.label}>Email</label>
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleInputChange}
                style={{
                  ...styles.input,
                  ...(errors.email ? styles.inputError : {})
                }}
                placeholder="Enter your email"
              />
              {errors.email && (
                <p style={styles.errorText}>{errors.email}</p>
              )}
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>Password</label>
              <input
                type="password"
                name="password"
                value={formData.password}
                onChange={handleInputChange}
                style={{
                  ...styles.input,
                  ...(errors.password ? styles.inputError : {})
                }}
                placeholder="Enter your password"
              />
              {errors.password && (
                <p style={styles.errorText}>{errors.password}</p>
              )}
            </div>

            {!isLogin && (
              <div style={styles.formGroup}>
                <label style={styles.label}>Confirm Password</label>
                <input
                  type="password"
                  name="confirmPassword"
                  value={formData.confirmPassword}
                  onChange={handleInputChange}
                  style={{
                    ...styles.input,
                    ...(errors.confirmPassword ? styles.inputError : {})
                  }}
                  placeholder="Confirm your password"
                />
                {errors.confirmPassword && (
                  <p style={styles.errorText}>{errors.confirmPassword}</p>
                )}
              </div>
            )}

            <button
              onClick={handleSubmit}
              disabled={isLoading}
              style={{
                ...styles.button,
                ...(isLoading ? styles.buttonDisabled : {})
              }}
              onMouseOver={(e) => {
                if (!isLoading) {
                  Object.assign(e.target.style, styles.buttonHover);
                }
              }}
              onMouseOut={(e) => {
                if (!isLoading) {
                  e.target.style.transform = 'none';
                  e.target.style.boxShadow = 'none';
                }
              }}
            >
              {isLoading ? (
                <div style={styles.loadingContent}>
                  <div style={styles.spinner}></div>
                  Processing...
                </div>
              ) : (
                isLogin ? 'Sign In' : 'Create Account'
              )}
            </button>

            <div style={styles.switchText}>
              {isLogin ? "Don't have an account? " : "Already have an account? "}
              <button
                onClick={switchMode}
                style={styles.switchButton}
                onMouseOver={(e) => e.target.style.color = '#5a67d8'}
                onMouseOut={(e) => e.target.style.color = '#667eea'}
              >
                {isLogin ? 'Sign up' : 'Sign in'}
              </button>
            </div>
          </div>
        </div>

        {/* Illustration Section */}
        <div style={styles.illustrationSection} className="illustration-section">
          <div>
            <div style={styles.illustrationIcon}>
              {isLogin ? '👋' : '🚀'}
            </div>
            <h3 style={styles.illustrationTitle}>
              {isLogin ? 'Hello Again!' : 'Start Your Journey'}
            </h3>
            <p style={styles.illustrationText}>
              {isLogin 
                ? 'We are happy to see you back. Sign in to continue where you left off.'
                : 'Create your account and unlock amazing features. Join thousands of satisfied users.'
              }
            </p>
            
            <div style={styles.featureList}>
              <div style={styles.feature}>
                <span>✓</span>
                <span>Secure & Protected</span>
              </div>
              <div style={styles.feature}>
                <span>✓</span>
                <span>Fast & Reliable</span>
              </div>
              <div style={styles.feature}>
                <span>✓</span>
                <span>24/7 Support</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AuthPage;